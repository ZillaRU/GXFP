import os
from argparse import ArgumentParser

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import Meter
# for AUC margin loss
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchsampler import ImbalancedDatasetSampler

from data_module.smiles_to_dglgraph import smiles_2_afpdgl
from model.encoder.afp import Intra_AttentiveFP
from model.predictor.non_linear import NonLinearPredictor
from utils import set_seed


# logger = logging.getLogger()


def collate_molgraphs(data):
    ids, smiles, graphs, labels = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    if labels[0] is not None:
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return ids, smiles, bg, labels
    else:
        return ids, smiles, bg, None


class CCKSDataset(Dataset):
    def __init__(self, df, smiles_to_graph, log_every=1000, isTest=False):
        self.isTest = isTest
        self.df = df
        self.smiles = self.df['SMILES'].tolist()
        self.ids = self.df['id'].tolist()
        self.labels = self.df['Y'].tolist() if not isTest else None
        self._pre_process(smiles_to_graph, log_every)

    def _pre_process(self, smiles_to_graph, log_every):
        print('Processing dgl graphs from scratch...')
        self.graphs = []
        for i, s in enumerate(self.smiles):
            if (i + 1) % log_every == 0:
                print('Processing molecule {:d}/{:d}'.format(i + 1, len(self)))
            self.graphs.append(smiles_to_graph(s))
        # Keep only valid molecules
        self.valid_ids = []
        graphs = []
        for i, g in enumerate(self.graphs):
            if g is not None:
                self.valid_ids.append(i)
                graphs.append(g)
            else:
                print(self.ids[i], self.smiles[i], 'is invalid')
        self.graphs = graphs
        self.smiles = [self.smiles[i] for i in self.valid_ids]
        self.ids = [self.ids[i] for i in self.valid_ids]
        self.labels = [self.labels[i] for i in self.valid_ids] if not self.isTest else None

    def get_labels(self):
        return self.labels  # torch.flatten(self.labels).tolist()

    def __getitem__(self, item):
        if self.isTest:
            return self.ids[item], self.smiles[item], self.graphs[item], None
        return self.ids[item], self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.smiles)


def run(train_loader, valid_loader, optimizer, criterion, model):
    device = args['device']

    def run_train_epoch():
        model.train()
        total_loss = 0
        err_loss_sum = 0
        diff_loss_sum = 0
        eval_meter = Meter()
        for batch_id, batch_data in enumerate(train_loader):
            ids, smiles, bg, labels = batch_data
            if len(smiles) == 1:
                continue
            bg, labels = bg.to(device), labels.to(device)
            logits, rep = model(bg)
            error_loss = criterion(logits, labels).mean()
            loss = (error_loss)
            eval_meter.update(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            err_loss_sum += error_loss.item()
            total_loss += loss.item()
            optimizer.step()

        return {
            'total_loss': total_loss,
            'BCE_loss': err_loss_sum,
            'diff_loss': 0,
            'AUROC': np.mean(eval_meter.compute_metric('roc_auc_score')),
            'AUPRC': np.mean(eval_meter.compute_metric('pr_auc_score'))
        }

    def run_eval_epoch():
        model.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(valid_loader):
                # print(batch_data)
                _, smiles, bg, labels = batch_data
                bg, labels = bg.to(device), labels.to(device)
                logits, _ = model(bg)
                # print(logits)
                eval_meter.update(logits, labels)
        return {
            'AUROC': np.mean(eval_meter.compute_metric('roc_auc_score')),
            'AUPRC': np.mean(eval_meter.compute_metric('pr_auc_score'))
        }

    def test_model():
        model.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(testloader):
                # ids, smiles, bg, labels
                id, smiles, bg, labels = batch_data
                bg, labels = bg.to(device), labels.to(device)
                logits, _ = model(bg)
                eval_meter.update(logits, labels)
        return {
            'AUROC': np.mean(eval_meter.compute_metric('roc_auc_score')),
            'AUPRC': np.mean(eval_meter.compute_metric('pr_auc_score'))
        }

    best_val_auc = 0.
    res_dict = None
    cnt_no_improve = 0
    dir = f'res/{dataset}'
    if not os.path.exists(dir):
        os.mkdir(dir)
    _dir = f'res/{dataset}/{n_cluster}'
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    __dir = f'{_dir}/{cluster_type}'
    if not os.path.exists(__dir):
        os.mkdir(__dir)
    ___dir = f'{__dir}/{n_cluster}'
    if not os.path.exists(___dir):
        os.mkdir(___dir)
    ____dir = f'{___dir}/AFP'
    if not os.path.exists(____dir):
        os.mkdir(____dir)
    with open(f'{____dir}/{split}.csv', 'w') as f:
        f.write('EPOCH,LOSS,BCE_loss,DIFF_loss,AUROC,AUPRC\n')
    for epoch_idx in range(args['epoch_num']):
        print(f"--------EPOCH {epoch_idx}---------")
        epoch_res = run_train_epoch()
        val_epoch_res = run_eval_epoch()
        with open(f'{____dir}/{split}.csv', 'a+') as f:
            f.write(
                f'{epoch_idx},{epoch_res["total_loss"]},{epoch_res["BCE_loss"]},{epoch_res["diff_loss"]},{val_epoch_res["AUROC"]},{val_epoch_res["AUPRC"]}\n')
        print(f"train AUROC: {epoch_res['AUROC']}, val AUROC: {val_epoch_res['AUROC']}")
        if val_epoch_res['AUROC'] > best_val_auc:
            best_val_auc = val_epoch_res['AUROC']
            res_dict = val_epoch_res
            cnt_no_improve = 0
        else:
            cnt_no_improve += 1
            if cnt_no_improve == args['patience']:
                break
    return res_dict, test_model()


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='_HIV')
    parser.add_argument('--split', type=str, default='5')
    parser.add_argument('--n_cluster', type=int, default=5)
    parser.add_argument('--cluster_type', type=str, default="ward")

    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nfold', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)

    # AUC-margin loss
    # parser.add_argument('--gamma', type=float, default=500)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--imratio', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.1)
    parser.add_argument('--predictor_hidden_feats', type=int, default=64)

    parser.add_argument('--epoch_num', type=int, default=2000)
    # parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    return parser.parse_args().__dict__


class MolPropModel(nn.Module):
    def __init__(self, args):
        super(MolPropModel, self).__init__()
        self.encoder = Intra_AttentiveFP(args['atom_feat_size'], args['edge_feat_size']).to(args['device'])
        self.predictor = NonLinearPredictor(args['graph_feat_size'], 1, args).to(args['device'])  # params

    def forward(self, bg):
        g_emb = self.encoder(bg)
        return self.predictor(g_emb), g_emb


def model_setup(atom_feat_size, edge_feat_size):
    args['device'] = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
    criterion = AUCMLoss(device=args['device'])
    # cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    args['atom_feat_size'] = atom_feat_size
    args['edge_feat_size'] = edge_feat_size
    args['graph_feat_size'] = 200
    model = MolPropModel(args)
    # optimizer = Adam([{"params": model.parameters()}],lr=args['lr'])
    optimizer = PESG(model,
                     a=criterion.a,
                     b=criterion.b,
                     alpha=criterion.alpha,
                     lr=args['lr'],
                     # gamma=args['gamma'],
                     margin=args['margin'],
                     # weight_decay=args['weight_decay'],
                     device=args['device']
                     )
    # return optimiz    er, cls_criterion, model
    return optimizer, criterion, model


args = get_args()

dataset, split, n_cluster, cluster_type = args['dataset'], args['split'], args['n_cluster'], args['cluster_type']
all_df = pd.read_csv(f'data/{dataset}/split/{split}/train.csv')
all_df.insert(all_df.shape[1], 'id', list(range(all_df.shape[0])))

test_df = pd.read_csv(f'data/{dataset}/split/{split}/test.csv')
test_df.insert(test_df.shape[1], 'id', list(range(test_df.shape[0])))

set_seed(args['seed'])

test_set = CCKSDataset(test_df, smiles_to_graph=smiles_2_afpdgl)
testloader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, drop_last=False,
                        collate_fn=collate_molgraphs)

kf = KFold(n_splits=args['nfold'], shuffle=True)

val_auc_list, val_aupr_list = [], []
test_auc_list, test_aupr_list = [], []

for train_index, test_index in kf.split(all_df):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_set = CCKSDataset(all_df.loc[train_index], smiles_to_graph=smiles_2_afpdgl)
    valid_set = CCKSDataset(all_df.loc[test_index], smiles_to_graph=smiles_2_afpdgl)
    train_dataloader = DataLoader(train_set,
                                  batch_size=args['batch_size'],
                                  shuffle=False,
                                  drop_last=True,
                                  sampler=ImbalancedDatasetSampler(dataset=train_set,
                                                                   labels=train_set.get_labels()),
                                  collate_fn=collate_molgraphs)
    valid_dataloader = DataLoader(valid_set,
                                  batch_size=args['batch_size'],
                                  shuffle=False,
                                  drop_last=False,
                                  collate_fn=collate_molgraphs)
    nsize = train_set[0][2].ndata['hv'].shape[1]
    esize = train_set[0][2].edata['he'].shape[1]
    opt, criterion, model = model_setup(nsize, esize)
    best_val, final_test_res = run(train_loader=train_dataloader,
                                   valid_loader=valid_dataloader,
                                   optimizer=opt,
                                   criterion=criterion,
                                   model=model)
    val_auc_list.append(best_val['AUROC'])
    val_aupr_list.append(best_val['AUPRC'])
    test_auc_list.append(final_test_res['AUROC'])
    test_aupr_list.append(final_test_res['AUPRC'])

if not os.path.exists(f'res/{dataset}/baseline_summary.csv'):
    with open(f'res/{dataset}/baseline_summary.csv', 'w') as f:
        f.write(f'Model,split,val_AUROC,val_AUPRC,test_AUROC,test_AUPRC\n')
with open(f'res/{dataset}/baseline_summary.csv', 'a+') as f:
    f.write(f'AttentiveFP,{split},{np.mean(val_auc_list)},{np.mean(val_aupr_list)},{np.mean(test_auc_list)},{np.mean(test_aupr_list)}\n')
