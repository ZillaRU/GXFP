from typing import List

import dgl
import torch
from dgllife.utils import ScaffoldSplitter, RandomSplitter
from torch.utils.data.dataloader import DataLoader
from torchsampler import ImbalancedDatasetSampler

from .load_datasets import CCKS22Test
from .smiles_to_dglgraph import smiles_2_dgl, smiles_2_kgdgl, smiles_2_afpdgl


def collate_molgraphs(data):
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


class DataModule(object):
    def __init__(self, encoder_name, data_name, num_workers: int, split_type: str, split_ratio: List, batch_size: int,
                 seed: int = 0) -> None:
        self.data_name = data_name
        if self.data_name in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', '_HIV', 'PCBA', 'Tox21']:
            self.task_type = 'classification'
        elif self.data_name in ['FreeSolv', 'Lipophilicity', 'ESOL']:
            self.task_type = 'regression'
        self.matrix = 'roc_auc_score'
        self.seed = seed

        self.num_workers = num_workers
        self.encoder_name = encoder_name
        self.dataset = self.load_dataset()
        self.split_type = split_type
        self.split_ratio = split_ratio
        # self.train_set, self.val_set, self.test_set = self.split_dataset()

        self.batch_size = batch_size
        self.task_num = self.dataset.n_tasks

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          sampler=ImbalancedDatasetSampler(dataset=self.dataset,
                                                           labels=self.dataset.get_labels()),
                          collate_fn=collate_molgraphs, num_workers=self.num_workers)

    def test_dataloader(self):
        def collate_test_molgraphs(data):
            smiles, graphs, ids = map(list, zip(*data))
            bg = dgl.batch(graphs)
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)
            return smiles, bg, ids
        return DataLoader(dataset=CCKS22Test(smiles_to_graph=smiles_2_kgdgl),
                          batch_size=self.batch_size,
                          collate_fn=collate_test_molgraphs,
                          num_workers=self.num_workers,
                          shuffle=False
                          )

    # def val_dataloader(self):
    #     return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collate_molgraphs,
    #                       num_workers=self.num_workers)

    def load_dataset(self):
        if self.encoder_name == 'GNN':
            if self.data_name == 'ccks22-kg':
                from .load_datasets import CCKS22
                dataset = CCKS22(smiles_2_kgdgl)
            elif self.data_name == 'ccks22':
                from .load_datasets import CCKS22
                dataset = CCKS22(smiles_2_dgl)
            else:
                raise ValueError('Unexpected dataset: {}'.format(self.data_name))

        elif self.encoder_name == 'MPNN' or self.encoder_name == 'KMPNN':
            if self.data_name == 'ccks22-kg':
                from .load_datasets import CCKS22
                dataset = CCKS22(smiles_2_kgdgl)
            elif self.data_name == 'ccks22':
                from .load_datasets import CCKS22
                dataset = CCKS22(smiles_2_dgl)
            else:
                raise ValueError('Unexpected dataset: {}'.format(self.data_name))
        elif self.encoder_name == 'AFP':
            from .load_datasets import CCKS22
            dataset = CCKS22(smiles_2_afpdgl)
            self.atom_feat_size = dataset[0][1].ndata['nfeats'].shape[1]
            self.edge_feat_size = dataset[0][1].ndata['efeats'].shape[1]
        return dataset

    def split_dataset(self):
        train_ratio, val_ratio, test_ratio = self.split_ratio
        if self.split_type == 'scaffold':
            train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
                self.dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
                scaffold_func='smiles')
        elif self.split_type == 'random':
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                self.dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio, random_state=self.seed)
        else:
            return ValueError("Expect the splitting method to be '', got {}".format(self.split_type))
        return train_set, val_set, test_set
