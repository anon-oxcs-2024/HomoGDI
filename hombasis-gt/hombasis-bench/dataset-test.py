import os
import numpy as np
import random

import torch
from torch.optim import Adam, AdamW
import torch_geometric.seed
from torch_geometric.loader import DataLoader
import wandb
from torch.utils.data import Subset

from data.get_data import load_zinc_homcount_dataset, load_zinc_subcount_dataset, load_zinc_subhom_dataset
from models.graph_reg import GINGraphReg, GCNGraphReg, GATGraphReg, MLPGraphReg
from zinc_utils import train, eval, parse_args

#NEW
import sys
sys.path.append("/data/coml-graphml/kell7068/thesis_code_1/GraphGPS/graphgps/loader")
from master_loader import load_dataset_master


if __name__ == "__main__":
    # set device
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # get args
    conf = parse_args()
    print(conf)

  ==============================")
  wandb.init(
        project=conf['project'],
        group=conf['group'],
        name=conf['model']
    )

    wandb.config.update(conf)
    
    default_seed = int(conf['seed'])
    print("==========================================================")
    print("Using device", str(device))
    print("Seed:", str(default_seed))
    print("======================== Args ===========================")
    print(conf)
    print("=====================
    # Set the seed for everything
    torch.manual_seed(default_seed)
    torch.cuda.manual_seed(default_seed)
    torch.cuda.manual_seed_all(default_seed)
    np.random.seed(default_seed)
    random.seed(default_seed)
    torch_geometric.seed.seed_everything(default_seed)
    
    print("Loading Data")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data', 'zinc-data')
    model_name = conf['model']
    batch_size = int(conf['batch_size'])
    use_counts = conf['use_counts']
    count_files = conf['count_files']
    count_type = conf['count_type']
    idx_list = conf['idx_list']
    hidden_dim = int(conf['hidden_dim'])
    description = conf['description']
    n_epochs = conf['epochs']
    dropout = conf['dropout'] if 'dropout' in conf else 'none'

    #new
    pe = conf['pe'] if 'pe' in conf else None

    
    if count_type == "homcounts" or count_type == "none":
        print('loading homcounts')
        train_data, val_data, test_data, count_dim = load_zinc_homcount_dataset(name='ZINC', hom_files=count_files, idx_list=idx_list, root=data_dir)
    elif count_type == "subcounts":
        sub_file = count_files[0]
        print('loading subcounts')
        train_data, val_data, test_data, count_dim = load_zinc_subcount_dataset(name='ZINC', sub_file=sub_file, idx_list=idx_list, root=data_dir)        
    elif count_type == "both":
        print('loading hom and subcounts')
        if "anchor" in count_files[0]:
            sub_file = 'zinc_3to10C_subgraph.json'
        else:
            sub_file = 'zinc_3to8C_multhom.json'
        train_data, val_data, test_data, count_dim = load_zinc_subhom_dataset(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root=data_dir)
    if pe != None and pe == 'RWSE': #ADD Functionality to change random walk length
        dataset = load_dataset_master('PyG-ZINC-Spasm')

        
    else:
        print('count type not supported')
    
    if not use_counts:
        count_files = 'none'
        count_dim = 0
        
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=["edge_attr"])
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, follow_batch=["edge_attr"])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, follow_batch=["edge_attr"])


    wandb.finish()