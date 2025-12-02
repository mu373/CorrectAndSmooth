import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob
from copy import deepcopy
from torch_geometric.utils import to_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import random
from outcome_correlation import *
from normalizers import DegreeNormalizer, PageRankNormalizer, DegreePageRankNormalizer
from adjacency import (
    StandardAdjacency, TwoHopAdjacency, SignlessLaplacian,
    Laplacian, KatzAdjacency
)
from cache import MatrixCache

def create_normalizer(args):
    """Create normalizer from CLI arguments."""
    if args.normalizer == 'degree':
        return DegreeNormalizer()
    elif args.normalizer == 'pagerank':
        return PageRankNormalizer(damping=args.pagerank_damping)
    elif args.normalizer == 'degree_pagerank':
        print(f"Creating DegreePageRankNormalizer with alpha={args.degree_pagerank_alpha}, damping={args.pagerank_damping}")
        return DegreePageRankNormalizer(
            alpha=args.degree_pagerank_alpha,
            damping=args.pagerank_damping
        )
    else:
        raise ValueError(f"Unknown normalizer: {args.normalizer}")


def create_adjacency(args):
    """Create adjacency transformer from CLI arguments."""
    if args.adjacency == 'standard':
        return StandardAdjacency()
    elif args.adjacency == '2hop':
        return TwoHopAdjacency()
    elif args.adjacency == 'signless_laplacian':
        return SignlessLaplacian()
    elif args.adjacency == 'laplacian':
        return Laplacian()
    elif args.adjacency == 'katz':
        return KatzAdjacency(beta=args.katz_beta, k=args.katz_k)
    else:
        raise ValueError(f"Unknown adjacency type: {args.adjacency}")


def main():
    parser = argparse.ArgumentParser(description='Outcome Correlations)')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)

    # Node weighting (normalization)
    parser.add_argument('--normalizer', type=str, default='degree',
                        choices=['degree', 'pagerank', 'degree_pagerank'],
                        help='Node weighting strategy (default: degree)')
    parser.add_argument('--pagerank-damping', type=float, default=0.85,
                        help='Damping factor for PageRank normalizer (default: 0.85)')
    parser.add_argument('--degree-pagerank-alpha', type=float, default=0.5,
                        help='Weight for degree in degree_pagerank normalizer (default: 0.5)')

    # Adjacency type
    parser.add_argument('--adjacency', type=str, default='standard',
                        choices=['standard', '2hop', 'signless_laplacian', 'laplacian', 'katz'],
                        help='Adjacency matrix type (default: standard)')
    parser.add_argument('--katz-beta', type=float, default=0.1,
                        help='Beta decay factor for Katz adjacency (default: 0.1)')
    parser.add_argument('--katz-k', type=int, default=3,
                        help='Number of hops for Katz adjacency (default: 3)')

    # Caching
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for caching matrices (default: None, no caching)')

    args = parser.parse_args()

    # Create normalizer and adjacency transformer
    normalizer = create_normalizer(args)
    adjacency = create_adjacency(args)

    # Create cache if specified
    cache = MatrixCache(args.cache_dir) if args.cache_dir else None

    print(f"Using normalizer: {normalizer.name}")
    print(f"Using adjacency: {adjacency.name}")
    if cache:
        print(f"Using cache: {args.cache_dir}")

    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')
    data = dataset[0]

    adj, norm_result = process_adj(data, normalizer=normalizer, adjacency=adjacency, cache=cache)
    normalized_adjs = gen_normalized_adjs(adj, norm_result)
    DAD, DA, AD = normalized_adjs
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    
    split_idx = dataset.get_idx_split()
  
    def eval_test(result, idx=split_idx['test']):
        return evaluator.eval({'y_true': data.y[idx],'y_pred': result[idx].argmax(dim=-1, keepdim=True),})['acc']
    
    if args.dataset == 'arxiv':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.9,
            'num_propagations': 50,
            'A': AD,
        }
        plain_dict = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859, 'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'} 
        gets you to 72.64
        """
        linear_dict = {
            'train_only': True,
            'alpha1': 0.98, 
            'alpha2': 0.65, 
            'A1': AD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523, 'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            'train_only': True,
            'alpha1': 0.9791632871592579, 
            'alpha2': 0.7564990804200602, 
            'A1': DA, 
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn = double_correlation_autoscale  
        
        gat_dict = {
            'labels': ['train'],
            'alpha': 0.8, 
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }
        gat_fn = only_outcome_correlation

        
    elif args.dataset == 'products':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.5,
            'num_propagations': 50,
            'A': DAD,
        }
        
        plain_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9, 
            'scale': 20.0, 
            'A1': DAD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        plain_fn = double_correlation_fixed
        
        linear_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9, 
            'scale': 20.0, 
            'A1': DAD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        linear_fn = double_correlation_fixed
        
        mlp_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.8, 
            'scale': 10.0, 
            'A1': DAD, 
            'A2': DA,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        mlp_fn = double_correlation_fixed




    model_outs = glob.glob(f'models/{args.dataset}_{args.method}/*.pt')
    
    if args.method == 'lp':
        out = label_propagation(data, split_idx, **lp_dict)
        print('Valid acc: ', eval_test(out, split_idx['valid']))
        print('Test acc:', eval_test(out, split_idx['test']))
        return
    
    get_orig_acc(data, eval_test, model_outs, split_idx)
    while True:
        if args.method == 'plain':
            evaluate_params(data, eval_test, model_outs, split_idx, plain_dict, fn = plain_fn)
        elif args.method == 'linear':
            evaluate_params(data, eval_test, model_outs, split_idx, linear_dict, fn = linear_fn)
        elif args.method == 'mlp':
            evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn = mlp_fn)
        elif args.method == 'gat':
            evaluate_params(data, eval_test, model_outs, split_idx, gat_dict, fn = gat_fn) 
#         import pdb; pdb.set_trace()
        break
        
#     name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
#     setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)
    
#     return

    
if __name__ == "__main__":
    main()
