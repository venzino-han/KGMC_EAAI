import math

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml, evaluate

from dataset import get_dataloader

from models.egmc import EGMC
from baselines.igmc import IGMC


def test(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)

    data_path = f'dataset/{args.dataset}/{args.dataset_filename}'
    efeat_path = f'dataset/{args.dataset}/{args.efeat_path}'
    _, _, test_loader =  get_dataloader(data_path, batch_size=args.batch_size, feature_path=efeat_path)

    model = EGMC(in_nfeats=args.in_nfeats, 
                        out_nfeats=args.out_nfeats,
                        in_efeats=args.in_efeats, 
                        out_efeats=args.out_efeats,
                        num_heads=args.num_heads,
                        review=args.review,
                        rating=args.rating,
                        timestamp=args.timestamp,
                        node_features=args.node_features,
                        )

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
        test_rmse = test(args, logger=logger)
        logger.info(f"DATASET : {args.dataset}")
        logger.info(f"Testing RMSE is {test_rmse:.6f}")

if __name__ == '__main__':
    main()