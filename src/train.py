import math, copy

import dgl
import pandas as pd
import numpy as np

import torch as th
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml, evaluate, feature_evaluate
from dataloader import get_graphs, get_dataloader

from models.kgmc import KGMC
from models.igmc import IGMC
from models.igmc_bert import IGMC_BERT

def train_epoch(model, loss_fn, optimizer, loader, device, logger, log_interval, train_model=None):
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        if train_model == 'IGMC_BERT':
            inputs = batch[0].to(device)
            vectors = batch[1].to(device)
            labels = batch[2].to(device)
            preds = model(inputs, vectors)
        elif train_model == 'KGMC':
            inputs = batch[0].to(device)
            keywords = batch[1].to(device)
            labels = batch[2].to(device)
            preds = model(inputs, keywords)
        else:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            preds = model(inputs)
        loss = loss_fn(preds, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)


NUM_WORKER = 8
def train(args:EasyDict, logger):
    th.manual_seed(0)
    np.random.seed(0)
    dgl.random.seed(0)

    data_path = f'data/{args.data_name}/{args.data_name}'
    if args.keywords is not None:
        with open(f'data/{args.data_name}/{args.keywords}', 'rb') as f:
            keyword_edge_matrix = np.load(f)
    else:
        keyword_edge_matrix = None

    # for bert embedding test
    if args.additional_feature is not None:
        n_side_features = 768*2
        with open(f'data/{args.data_name}/{args.additional_feature}', 'rb') as f:
            additional_feature = np.load(f)
    else:
        additional_feature = None
        n_side_features = 0

    train_graph, valid_graph, test_graph = get_graphs(data_path=data_path)
    
    keyword_edge_k = args.keyword_edge_k
    train_loader = get_dataloader(train_graph, keyword_edge_cooc_matrix=keyword_edge_matrix, 
                                 keyword_edge_k=keyword_edge_k,
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER,
                                 additional_feature=additional_feature,
                                 shuffle=True, 
                                 )
    valid_loader = get_dataloader(valid_graph, keyword_edge_cooc_matrix=keyword_edge_matrix, 
                                 keyword_edge_k=keyword_edge_k,
                                 batch_size=args.batch_size, 
                                 num_workers=NUM_WORKER, 
                                 additional_feature=additional_feature,
                                 shuffle=False,
                                 )
    test_loader = get_dataloader(test_graph, keyword_edge_cooc_matrix=keyword_edge_matrix, 
                                 batch_size=args.batch_size, 
                                 keyword_edge_k=keyword_edge_k,
                                 num_workers=NUM_WORKER, 
                                 additional_feature=additional_feature, 
                                 shuffle=False,
                                 )

    ### prepare data and set model
    in_feats = (args.hop+1)*2 
    if args.model_type == 'IGMC':
        model = IGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=args.num_relations, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)

    elif args.model_type == 'IGMC_BERT':
        model = IGMC_BERT(in_feats=in_feats, 
                        latent_dim=args.latent_dims,
                        num_relations=args.num_relations, 
                        num_bases=4, 
                        regression=True,
                        side_features=True,
                        n_side_features=n_side_features,
                        edge_dropout=args.edge_dropout,
                        ).to(args.device)

    elif args.model_type == 'KGMC':
        model = KGMC(in_feats=in_feats, 
                     latent_dim=args.latent_dims,
                     num_relations=5, 
                     num_bases=4, 
                     regression=True,
                     edge_dropout=args.edge_dropout,
                     ).to(args.device)
    
    elif args.model_type == 'EGMC':
        pass
    if args.parameters is not None:
        model.load_state_dict(th.load(f"./parameters/{args.parameters}"))
        
    loss_fn = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    
    best_epoch = 0
    best_rmse = np.inf

    logger.info(f"Start training ... learning rate : {args.train_lr}")
    epochs = list(range(1, args.train_epochs+1))

    eval_func_map = {
        'KGMC': feature_evaluate,
        'IGMC_BERT': feature_evaluate,
        'IGMC': evaluate,
    }
    eval_func = eval_func_map.get(args.model_type, evaluate)
    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
    
        train_loss = train_epoch(model, loss_fn, optimizer, train_loader, 
                                 args.device, logger, args.log_interval, train_model=args.model_type)
        val_rmse = eval_func(model, valid_loader, args.device)
        test_rmse = eval_func(model, test_loader, args.device)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': train_loss,
            'val_rmse' : val_rmse,
            'test_rmse': test_rmse,
        }
        logger.info('=== Epoch {}, train loss {:.6f}, val rmse {:.6f}, test rmse {:.6f} ==='.format(*eval_info.values()))

        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_rmse > test_rmse:
            logger.info(f'new best test rmse {test_rmse:.6f} ===')
            best_epoch = epoch_idx
            best_rmse = test_rmse
            best_state = copy.deepcopy(model.state_dict())

    th.save(best_state, f'./parameters/{args.key}_{args.data_name}_{best_rmse:.4f}.pt')
    logger.info(f"Training ends. The best testing rmse is {best_rmse:.6f} at epoch {best_epoch}")
    return best_rmse
    
import yaml
from collections import defaultdict
from datetime import datetime

def main():
    with open('./train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        date_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')
        for k,v in args.items():
            logger.info(f'{k}: {v}')

        test_results = defaultdict(list)
        best_lr = None
        for data_name in args.datasets:
            sub_args = args
            sub_args['data_name'] = data_name
            best_rmse_list = []
            for lr in args.train_lrs:
                sub_args['train_lr'] = lr
                best_rmse = train(sub_args, logger=logger)
                test_results[data_name].append(best_rmse)
                best_rmse_list.append(best_rmse)
            
            logger.info(f"**********The final best testing RMSE is {min(best_rmse_list):.6f} at lr {best_lr}********")
            logger.info(f"**********The mean testing RMSE is {np.mean(best_rmse_list):.6f}, {np.std(best_rmse_list)} ********")
        
            mean_std_dict = dict()
            for dataset, results in test_results.items():
                mean_std_dict[dataset] = [f'{np.mean(results):.4f} ± {np.std(results):.5f}']
            mean_std_df = pd.DataFrame(mean_std_dict)
            mean_std_df.to_csv(f'./results/{args.key}_{date_time}.csv')
        
if __name__ == '__main__':
    main()