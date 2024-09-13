import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp

import sys
sys.path.append('/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/qm9/src/utils')
import dataset_loader as dl
from shortest_paths import ShortestPathTransform

# sys.path.append('/data/coml-graphml/kell7068/thesis_code_1/GraphGPS/graphgps/loader')
# from master_loader import join_dataset_splits

import time

sys.path.append('/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/qm9/data_GraphGym_QM9')
from CustomDataset import CustomDataset

#copied from /data/coml-graphml/kell7068/thesis_code_1/GraphGPS/graphgps/loader/master_loader.py to avert double register
def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])

    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

# class CustomDataset(InMemoryDataset):
#     def __init__(self,listOfDataObjects, root=None):
#         super().__init__(root=root)
#         self.data, self.slices = self.collate(listOfDataObjects)
    
#     def __len__(self):
#         return len(self.slices)
    
#     def __getitem__(self, idx):
#         sample = self.get(idx)
#         return sample

#takes in three lists of pyg elements and returns a list of three pyg InMemoryDatasets
def get_qm9_datasets(tr_graphs, val_graphs, tst_graphs):
    data_dir = "/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/qm9/data_GraphGym_QM9/storage"
    tr_graphs = CustomDataset(tr_graphs, root=data_dir)
    val_graphs = CustomDataset(val_graphs, root=data_dir)
    tst_graphs = CustomDataset(tst_graphs, root=data_dir)
    return [tr_graphs, val_graphs, tst_graphs]

def get_qm9_graphHC_lists(root_dir, include_shortest_path=True):

    dataset_path = osp.join(root_dir, "data", "QM9")
    if include_shortest_path:
        transform_class = ShortestPathTransform(max_distance=10)
    else:
        transform_class = None

    qm9_proc_root = osp.join(dataset_path, "QM9_proc")
    tr_graphs = dl.read_qm9(
        dataset_path, "train", transform_class.transform, qm9_proc_root
    )
    val_graphs = dl.read_qm9(
        dataset_path, "valid", transform_class.transform, qm9_proc_root
    )
    tst_graphs = dl.read_qm9(
        dataset_path, "test", transform_class.transform, qm9_proc_root
    )
    num_feat = 15  # + 30 dim of all-5 vertex homcounts + 1 6 cycle
    num_pred = 13  # Class here really is used in the sense of
    return tr_graphs, val_graphs, tst_graphs, num_feat, num_pred

start_t = time.time()
tr_graphs, val_graphs, tst_graphs, num_feat, num_pred = get_qm9_graphHC_lists('/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/qm9')
get_qm9_graphHC_lists_t = time.time()
print(f'len tr_graphs: {len(tr_graphs)}')
print(f'len val_graphs: {len(val_graphs)}')
print(f'len tst_graphs: {len(tst_graphs)}')
#print(f'tr_graphs list: {tr_graphs}')
print(f'tr_graphs list [0]:{tr_graphs[0]}')
dataset_lists = get_qm9_datasets(tr_graphs, val_graphs, tst_graphs)
get_qm9_datasets_t = time.time()
print(f'len tr_dataset: {len(dataset_lists[0])}')
print(f'dataset_lists: {dataset_lists}')
print(f'dataset_lists[0]: {dataset_lists[0]}')
print(f'dataset_lists[0][0]: {dataset_lists[0][0]}')
joined_dataset = join_dataset_splits([dataset_lists[0], dataset_lists[1], dataset_lists[2]])
joined_dataset_t = time.time()
print(f'dataset slices: {joined_dataset.slices}')
print(f'dataset data: {joined_dataset.data}')

#UNCOMMENT BELOW TO SAVE GRAPH_HC
torch.save(joined_dataset,"/data/coml-graphml/kell7068/thesis_code_1/datasets/QM9-GraphHC/processed/joined.pt")
save_t = time.time()

print(f'joined dataset: {joined_dataset}')
print(f'joineddataset[0]: {joined_dataset[0]}')
print(f'time to complete get_qm9_graphHC_lists: {get_qm9_graphHC_lists_t-start_t}')
print(f'time to complete get_qm9_datasets: {get_qm9_datasets_t-get_qm9_graphHC_lists_t}')
print(F'time to complete joining the datasets: {joined_dataset_t-get_qm9_datasets_t}')
print(f'time to save: {save_t-joined_dataset_t}')
print(f'total time: {save_t-start_t}')
print('(time unit is seconds I believe)')

# def preformat_ZINC(dataset_dir, name, postfix=None):
#     """Load and preformat ZINC datasets.

#     Args:
#         dataset_dir: path where to store the cached dataset
#         name: select 'subset' or 'full' version of ZINC

#     Returns:
#         PyG dataset object
#     """
#     data_dir = "/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/hombasis-bench/data/zinc-data"
#     dataset = None
#     if name not in ['subset', 'full']:
#         raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
#     dataset = join_dataset_splits(
#         [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
#         for split in ['train', 'val', 'test']]
#     )
#     if postfix != None and "Spasm" in postfix and name == 'subset':
#         count_files = ['zinc_with_homs_c7.json', 'zinc_with_homs_c8.json']
#         idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 15, 20, 21, 22, 24, 25, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
#         sub_file = 'zinc_3to8C_multhom.json'
#         dataset = get_data.add_zinc_subhom(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root=data_dir, dataset=dataset)

#     return dataset