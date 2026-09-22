"""Epoch-boundary checkpoints for the otherwise unchanged Zoology training recipe."""
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from zoology.train import Trainer


class ResumableTrainer(Trainer):
    def fit(self):
        folder = Path(os.environ["MQAR_RUN_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"w{os.environ['ZOO_DM']}-d{os.environ['ZOO_DS']}.pt"
        self.model.to(self.device)
        with torch.no_grad():
            self.model(next(iter(self.train_dataloader))[0].to(self.device))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        parameters = (self.optimizer_parameters() if hasattr(self, 'optimizer_parameters')
                      else self.model.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0)
        start = 0
        if path.exists():
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            start = ck["next_epoch"]
            for name, module in self.model.named_modules():
                if hasattr(module, "_steps"):
                    module._steps = ck.get("module_steps", {}).get(
                        name, 1 + start * len(self.train_dataloader))
            resume_rejected = (os.environ.get('MQAR_RESUME_REJECTED') == '1'
                               and self.early_stopping_metric is None
                               and start < self.max_epochs
                               and ck['metrics'].get('gate/failed', 0) > .5)
            if ck["complete"] and not resume_rejected:
                print(f"ALREADY COMPLETE: {path}", flush=True); return
            if resume_rejected:
                # The stopping epoch saved before its scheduler step. Complete
                # that step so continuation follows the original32-epoch curve.
                self.scheduler.step()
            torch.set_rng_state(ck["rng_cpu"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in ck["rng_cuda"]])
            np.random.set_state(ck["rng_numpy"]); random.setstate(ck["rng_python"])
            print(f"RESUMED {path.name} at epoch {start}", flush=True)
        print(f"TRAINING {path.name} parameter_tensors={len(list(self.model.parameters()))} epochs={self.max_epochs}", flush=True)
        started = time.monotonic()
        for epoch in range(start, self.max_epochs):
            self.train_epoch(epoch)
            metrics = self.test(epoch)
            early = (self.early_stopping_metric is not None and
                     metrics[self.early_stopping_metric] > self.early_stopping_threshold)
            complete = early or epoch + 1 == self.max_epochs
            if not early:
                self.scheduler.step()
            checkpoint = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(), next_epoch=epoch + 1, complete=complete,
                module_steps={name: module._steps for name, module in self.model.named_modules()
                              if hasattr(module, "_steps")},
                metrics=metrics, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                rng_numpy=np.random.get_state(), rng_python=random.getstate())
            temp = path.with_suffix(".tmp")
            torch.save(checkpoint, temp); os.replace(temp, path)
            status = dict(next_epoch=epoch + 1, complete=bool(complete), metrics=metrics)
            path.with_suffix(".json").write_text(json.dumps(status, default=float, indent=2))
            print(f"CHECKPOINT epoch={epoch + 1} complete={complete} {path}", flush=True)
            if complete:
                return
            if time.monotonic() - started > float(os.environ.get("MQAR_MAX_MINUTES", "100")) * 60:
                print("TIME LIMIT: checkpoint saved", flush=True); return
