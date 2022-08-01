import math

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml, evaluate

from dataloader import get_dataloader, get_graphs

# from models.egmc import EGMC
from models.igmc import IGMC
from models.kgmc import KGMC


def test(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)

    data_path = f'data/{args.data_name}/{args.data_name}'
    # efeat_path = f'dataset/{args.datasets}/{args.additional_feature}'
    if args.keywords is not None:
        with open(f'data/{args.data_name}/{args.keywords}', 'rb') as f:
            keyword_edge_matrix = np.load(f)
    else:
        keyword_edge_matrix = None
    train_graph, valid_graph, test_graph = get_graphs(data_path=data_path)

    test_loader = get_dataloader(test_graph, keyword_edge_cooc_matrix=keyword_edge_matrix, 
                                 batch_size=args.batch_size, 
                                 keyword_edge_k=100,
                                 num_workers=16, 
                                 additional_feature=None, 
                                 shuffle=False,
                                 )

    if args.model_type == 'IGMC':
        model = IGMC(in_feats=4, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    elif args.model_type == 'KGMC':
        model = KGMC(in_feats=4, 
                     latent_dim=args.latent_dims,
                     num_relations=8, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)
    

    model.load_state_dict(th.load(f"./parameters/{args.parameters}"))
    model.to(args.device)

    logger.info("Loading network finished ...\n")
    test_rmse = evaluate(model, test_loader, args.device)
    return test_rmse
    
import yaml

def main():
    with open('./test_configs/test_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        for data_name in args.datasets:
            sub_args = args
            sub_args['data_name'] = data_name
            best_rmse_list = []
            test_rmse = test(args, logger=logger)
            logger.info(f"DATASET : {args.datasets}")
            logger.info(f"Testing RMSE is {test_rmse:.6f}")

if __name__ == '__main__':
    main()