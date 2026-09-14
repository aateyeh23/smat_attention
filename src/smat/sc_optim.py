"""AdamW groups for a separate G-write learning-rate multiplier."""
def write_gate_groups(model, weight_decay, gate_lr_scale):
    groups = {}
    for name, parameter in model.named_parameters():
        decay = 0.0 if (parameter.ndim < 2 or 'A_log' in name or
                       'dt_bias' in name or name.endswith('.D')) else weight_decay
        scale = gate_lr_scale if '.g_write_proj.' in name else 1.0
        groups.setdefault((decay, scale), []).append(parameter)
    return [dict(params=params, weight_decay=decay, lr_scale=scale)
            for (decay, scale), params in groups.items()]
