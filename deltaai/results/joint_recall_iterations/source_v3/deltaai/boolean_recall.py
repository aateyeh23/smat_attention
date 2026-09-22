"""Matched retrieval, local AND, and retrieval+AND with a balanced truth table."""
import argparse
import gc
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import torch
from synthetic_memory import run

PAD, QUERY, ANSWER_A, ANSWER_B = 0, 1, 2, 3
ID_BASE, ID_COUNT, BIT_BASE, VOCAB = 4, 16, 20, 22


class BooleanRecall:
    def __init__(self, condition, length=64):
        self.condition, self.length = condition, length

    def task_tokens(self, task):
        return ID_COUNT, BIT_BASE, VOCAB

    def recipe_metadata(self):
        return dict(condition=self.condition, values=2, chance=.5,
                    metric='full-vocabulary accuracy; exact answers and all four truth cases',
                    truth_table='00/01/10/11 equally frequent in each batch; shuffled example order',
                    constant_zero_accuracy=.5 if self.condition=='retrieval' else .75,
                    query_format='QUERY keyA keyB ANSWER_A ANSWER_B; direct AND substitutes supplied bits for keys')

    def sequence_length(self, task, load, *unused):
        if not 2 <= load <= ID_COUNT or 2*load+5 > self.length or self.length==192:
            raise ValueError('Two distinct query keys and all records must fit the common length')
        return self.length

    def generate(self, task, load, batch, rng, *unused):
        length=self.sequence_length(task,load)
        x=np.zeros((batch,length),dtype=np.int64)
        pair_bits=np.empty((batch,2),dtype=np.int64)
        cases=rng.permutation(np.arange(batch)%4)
        for b,case in enumerate(cases):
            keys=rng.choice(ID_COUNT,load,replace=False)+ID_BASE
            bits=rng.integers(2,size=load)
            query=rng.choice(load,2,replace=False)
            pair_bits[b]=[case//2,case%2]
            bits[query]=pair_bits[b]
            order=rng.permutation(load)
            slots=np.sort(rng.choice((length-5)//2,load,replace=False))*2
            x[b,slots]=keys[order]; x[b,slots+1]=bits[order]+BIT_BASE
            q=pair_bits[b]+BIT_BASE if self.condition=='direct_and' else keys[query]
            x[b,-5:]=[QUERY,*q,ANSWER_A,ANSWER_B]
        if self.condition=='retrieval':
            positions=np.tile([length-2,length-1],(batch,1))
            targets=pair_bits+BIT_BASE
        else:
            positions=np.full((batch,1),length-1,dtype=np.int64)
            targets=pair_bits.all(axis=1).astype(np.int64)[:,None]+BIT_BASE
        return x,positions,targets

    def decode(self,data):
        pairs, tables=[],[]
        for tokens in data[0]:
            table={int(tokens[j]):int(tokens[j+1]-BIT_BASE)
                   for j in range(0,self.length-5-1,2) if tokens[j]!=PAD}
            pair=tokens[-4:-2]-BIT_BASE if self.condition=='direct_and' else [table[int(k)] for k in tokens[-4:-2]]
            pairs.append(pair);tables.append(table)
        return np.asarray(pairs),tables

    def query_blind_controls(self, task, data):
        _,tables=self.decode(data)
        guesses=[]
        for table in tables:
            n=len(table);ones=sum(table.values())
            probability=ones/n if self.condition=='retrieval' else ones*(ones-1)/(n*(n-1))
            guesses.append(BIT_BASE+int(probability>.5))
        return dict(always_zero_accuracy=float((data[2]==BIT_BASE).mean()),
                    query_blind_count_oracle_accuracy=float((np.asarray(guesses)[:,None]==data[2]).mean()))

    def binding_control(self, task, data, predictions):
        pairs,tables=self.decode(data)
        correct=(predictions==data[2])
        case=2*pairs[:,0]+pairs[:,1]
        result={}
        for c in range(4):
            selected=case==c
            result[f'truth_{c:02b}_accuracy']=float(correct[selected].mean())
            result[f'truth_{c:02b}_exact_accuracy']=float(correct[selected].all(axis=1).mean())
            result[f'truth_{c:02b}_examples']=int(selected.sum())
        result['worst_truth_exact_accuracy']=min(result[f'truth_{c:02b}_exact_accuracy'] for c in range(4))
        if self.condition=='retrieval':
            valid=((predictions==BIT_BASE)|(predictions==BIT_BASE+1)).all(axis=1)
            composed=(predictions==BIT_BASE+1).all(axis=1)
            result['composed_and_accuracy']=float((valid&(composed==pairs.all(axis=1))).mean())
            other=[]
            for tokens,table,pred in zip(data[0],tables,predictions):
                for key,answer in zip(tokens[-4:-2],pred):
                    other.append(np.mean([answer==BIT_BASE+v for k,v in table.items() if k!=key]))
        else:
            positive=data[2]==BIT_BASE+1
            result['balanced_accuracy']=float((correct[positive].mean()+correct[~positive].mean())/2)
            other=[]
            for pair,table,pred in zip(pairs,tables,predictions[:,0]):
                alternatives=[int(a and b)+BIT_BASE for a,b in combinations(table.values(),2)]
                # Remove the queried pair's outcome; every unordered key pair
                # occurs once, even when multiple pairs share an outcome.
                alternatives.remove(int(pair.all())+BIT_BASE)
                other.append(np.mean(np.asarray(alternatives)==pred) if alternatives else float('nan'))
        result['binding_gap_pp']=100*(float(correct.mean())-float(np.mean(other))) if len(next(iter(tables)))>2 or self.condition=='retrieval' else None
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--conditions',default='direct_and,retrieval,retrieval_and')
    p.add_argument('--records',default='4')
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--families',default='mamba2,gdn_current')
    p.add_argument('--ds',default='1,3')
    p.add_argument('--seeds',default='123')
    p.add_argument('--max-minutes',type=float,default=13)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'results/boolean_recall')
    a=p.parse_args();started=time.monotonic()
    for records in map(int,a.records.split(',')):
        for condition in a.conditions.split(','):
            if condition not in ['retrieval','direct_and','retrieval_and']: raise ValueError(condition)
            task=BooleanRecall(condition,a.length);task.sequence_length('boolean',records)
            for seed in map(int,a.seeds.split(',')):
                for d in map(int,a.ds.split(',')):
                    for family in a.families.split(','):
                        folder=a.root/f'{condition}-w{a.width}-r{records}-t{a.length}-{family}-d{d}-s{seed}'
                        if (folder/'result.json').exists(): continue
                        remaining=a.max_minutes-(time.monotonic()-started)/60
                        if remaining<2:return
                        args=argparse.Namespace(task='boolean',condition=condition,load=records,family=family,d=d,width=a.width,
                            seed=seed,steps=500,batch=32,lr=.01,min_length=a.length,updates=1,hops=0,eval_every=100,
                            eval_examples=1024,test_examples=4096,max_minutes=remaining,out=str(folder))
                        run(args,task_api=task)
                        gc.collect();torch.cuda.empty_cache()
                        if not (folder/'result.json').exists():return
    print('CAMPAIGN_COMPLETE',flush=True)


if __name__=='__main__':main()
