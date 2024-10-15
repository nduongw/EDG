from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
import argparse
import collections
import json
import os
import random
import sys
import time
import math

parser = argparse.ArgumentParser(description='Ensemble Domain generalization')
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--dataset', type=str)
parser.add_argument('--train_envs', type=int, nargs='+', default=[])
parser.add_argument('--test_envs', type=int, nargs='+', default=[])
parser.add_argument('--algorithm', type=str, default="ERM")
parser.add_argument('--hparams_seed', type=int, default=0,
    help='Seed for random hparams (0 means "default hparams")')
parser.add_argument('--trial_seed', type=int, default=0,
    help='Trial number (used for seeding split_dataset and '
    'random_hparams).')
parser.add_argument('--seed', type=int, default=0,
    help='Seed for everything else')
parser.add_argument('--holdout_fraction', type=float, default=0.2)

# New args for ratatouille
parser.add_argument('--what_is_trainable', type=str, default="all")
parser.add_argument('--path_init', type=str, default="")
parser.add_argument('--aux_dir', type=str, default="")
parser.add_argument('--fusing_range', type=float, default=-1)
args = parser.parse_args(raw_args)

if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
else:
    hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
        misc.seed_hash(args.hparams_seed, args.trial_seed))

dataset = vars(datasets)[args.dataset](args.data_dir, args.train_envs, args.test_envs, hparams)

# Split each env into an 'in-split' and an 'out-split'. We'll train on
# each in-split except the test envs, and evaluate on all splits.
in_splits = []
out_splits = []

for env_i, env in enumerate(dataset):
    out, in_ = misc.split_dataset(env,
        int(len(env)*args.holdout_fraction),
        misc.seed_hash(args.trial_seed, env_i))
    in_splits.append(in_)
    out_splits.append(out)

train_loaders = [InfiniteDataLoader(
    dataset=env,
    weights=None,
    batch_size=hparams['batch_size'],
    num_workers=dataset.N_WORKERS)
    for i, env in enumerate(in_splits)
    if (i not in args.test_envs) and (i in args.train_envs)]

eval_loaders = []
for i in range(len(in_splits)):
    if i in args.train_envs or i in args.test_envs:
        eval_loaders.append(FastDataLoader(dataset=in_splits[i-1], batch_size=64, num_workers=dataset.N_WORKERS))
        eval_loaders.append(FastDataLoader(dataset=out_splits[i-1], batch_size=64, num_workers=dataset.N_WORKERS))


