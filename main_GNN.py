import os
import random

# for reproduction
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling

from model import Set2Set

random.seed(7)

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

from data_module.smiles_to_dglgraph import smiles_2_afpdgl, smiles_2_Weave_dgl, smiles_2_basedgl
from dgllife.model import MPNNGNN, WeaveGNN, WeaveGather, AttentiveFPGNN, \
    AttentiveFPReadout, WeightedSumAndMax, MLPPredictor, GAT, GCN, GIN
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


class MolDataset(Dataset):
    def __init__(self, df, smiles_to_graph, log_every=1000, isTest=False):
        self.isTest = isTest
        self.df = df
        self.smiles = self.df['SMILES'].tolist()
        self.ids = self.df['id'].tolist()
        self.labels = self.df[args['label']].tolist() if not isTest else None
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


def run(train_loader, valid_loader, optimizer, criterion, model, pos_dict, neg_dict, alpha=0.01):
    device = args['device']
    diff_loss_fn = nn.TripletMarginLoss(margin=1, p=2)

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
            if args['backbone'] == 'GCN' or args['backbone'] == 'GAT':
                logits, rep = model(bg, bg.ndata['hv'].to(torch.float32))
            else:
                logits, rep = model(bg, bg.ndata['hv'].to(torch.float32), bg.edata['he'].to(torch.float32))
            error_loss = criterion(logits, labels).mean()
            if args['alpha'] > 0:
                id2nid = dict(zip(ids, range(len(ids))))
                batch_anchors, batch_ps, batch_ns = [], [], []
                for ci in range(n_cluster):
                    neg_batch_ci = []
                    for i in neg_dict[ci]:
                        if i in id2nid:
                            neg_batch_ci.append(id2nid[i])
                    if len(neg_batch_ci) == 0:
                        continue
                    for cj in range(n_cluster):
                        if ci != cj:
                            for anch in pos_dict[ci]:
                                if anch not in id2nid:
                                    continue
                                cnt = 0
                                for p in pos_dict[cj]:
                                    if p not in id2nid:
                                        continue
                                    batch_anchors.append(id2nid[anch])
                                    batch_ps.append(id2nid[p])
                                    cnt += 1
                                batch_ns += random.choices(neg_batch_ci, k=cnt)
                diff_loss = torch.sum(diff_loss_fn(rep[batch_anchors], rep[batch_ps], rep[batch_ns]))
                diff_loss_sum += diff_loss.item()
                print(diff_loss.item())
            else:
                diff_loss = 0
            loss = (error_loss + alpha * diff_loss)
            eval_meter.update(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            err_loss_sum += error_loss.item()
            total_loss += loss.item()
            optimizer.step()

        return {
            'total_loss': total_loss,
            'BCE_loss': err_loss_sum,
            'diff_loss': diff_loss_sum,
            'AUROC': np.mean(eval_meter.compute_metric('roc_auc_score')),
            'AUPRC': np.mean(eval_meter.compute_metric('pr_auc_score'))
        }

    def run_eval_epoch():
        model.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(valid_loader):
                _, smiles, bg, labels = batch_data
                bg, labels = bg.to(device), labels.to(device)
                if args['backbone'] == 'GCN' or args['backbone'] == 'GAT':
                    logits, _ = model(bg, bg.ndata['hv'].to(torch.float32))
                else:
                    logits, _ = model(bg, bg.ndata['hv'].to(torch.float32), bg.edata['he'].to(torch.float32))
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
                if args['backbone'] == 'GCN' or args['backbone'] == 'GAT':
                    logits, rep = model(bg, bg.ndata['hv'].to(torch.float32))
                else:
                    logits, rep = model(bg, bg.ndata['hv'].to(torch.float32), bg.edata['he'].to(torch.float32))
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
    ____dir = f'{___dir}/GXFP_{args["backbone"]}'
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
        elif epoch_idx >= (100 if args['dataset'] == '_HIV' else 500):
            cnt_no_improve += 1
            if epoch_idx >= (100 if args['dataset'] == '_HIV' else 500) and cnt_no_improve == args['patience']:
                break
    return res_dict, test_model()


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--split', type=str, default='7')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--n_cluster', type=int, default=40)
    parser.add_argument('--cluster_type', type=str, default="ward")
    parser.add_argument('--label', type=str, required=True)

    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)

    # AUC-margin loss
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--imratio', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.1)
    parser.add_argument('--predictor_hidden_feats', type=int, default=64)

    parser.add_argument('--epoch_num', type=int, default=2000)
    return parser.parse_args().__dict__


class WeavePredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=200,
                 gnn_activation=torch.nn.functional.relu,
                 graph_feats=200,
                 gaussian_expand=True,
                 gaussian_memberships=None,
                 readout_activation=nn.Tanh(),
                 n_tasks=1):
        super(WeavePredictor, self).__init__()

        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)
        self.node_to_graph = nn.Sequential(
            nn.Linear(gnn_hidden_feats, graph_feats),
            readout_activation,
            nn.BatchNorm1d(graph_feats)
        )
        self.readout = WeaveGather(node_in_feats=graph_feats,
                                   gaussian_expand=gaussian_expand,
                                   gaussian_memberships=gaussian_memberships,
                                   activation=readout_activation)
        self.predict = nn.Linear(graph_feats, n_tasks)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        node_feats = self.node_to_graph(node_feats)
        g_feats = self.readout(g, node_feats)

        return self.predict(g_feats), g_feats


class MPNNPredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=200,
                 edge_hidden_feats=200,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats), graph_feats


class GINPredictor(nn.Module):
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1):
        super(GINPredictor, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(num_node_emb_list=num_node_emb_list,
                       num_edge_emb_list=num_edge_emb_list,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       JK=JK,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max' or 'attention', got {}".format(readout))

        if JK == 'concat':
            self.predict = nn.Linear((num_layers + 1) * emb_dim, n_tasks)
        else:
            self.predict = nn.Linear(emb_dim, n_tasks)

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        node_feats = self.gnn(g, categorical_node_feats, categorical_edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats), graph_feats


class AttentiveFPPredictor(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 n_tasks=1,
                 dropout=0.):
        super(AttentiveFPPredictor, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats), g_feats


class GATPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(GATPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats), graph_feats


class GCNPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, activation=None, residual=None, batchnorm=None,
                 dropout=None, classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(GCNPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats), graph_feats


def model_setup(atom_feat_size, edge_feat_size):
    args['device'] = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
    criterion = AUCMLoss(device=args['device'])
    args['atom_feat_size'] = atom_feat_size
    args['edge_feat_size'] = edge_feat_size
    args['graph_feat_size'] = 200
    if args['backbone'] == 'MPNN':
        model = MPNNPredictor(args['atom_feat_size'], args['edge_feat_size']).to(args['device'])
    elif args['backbone'] == 'Weave':
        model = WeavePredictor(args['atom_feat_size'], args['edge_feat_size']).to(args['device'])
    elif args['backbone'] == 'GAT':
        model = GATPredictor(args['atom_feat_size'], [200, 200]).to(args['device'])
    elif args['backbone'] == 'GCN':
        model = GCNPredictor(args['atom_feat_size'], [200, 200]).to(args['device'])
    elif args['backbone'] == 'GIN':
        model = GINPredictor([args['atom_feat_size']], [args['edge_feat_size']]).to(args['device'])
    elif args['backbone'] == 'AFP':
        model = AttentiveFPPredictor(args['atom_feat_size'], args['edge_feat_size']).to(args['device'])
    else:
        raise NotImplementedError
    optimizer = PESG(model,
                     a=criterion.a,
                     b=criterion.b,
                     alpha=criterion.alpha,
                     lr=args['lr'],
                     margin=args['margin'],
                     device=args['device']
                     )
    return optimizer, criterion, model


args = get_args()
dataset, split, n_cluster, cluster_type = args['dataset'], args['split'], args['n_cluster'], args['cluster_type']
all_df = pd.read_csv(f'data/{dataset}/scaffold/{split}/train.csv')
all_df.insert(all_df.shape[1], 'id', list(range(all_df.shape[0])))

test_df = pd.read_csv(f'data/{dataset}/scaffold/{split}/test.csv')
test_df.insert(test_df.shape[1], 'id', list(range(test_df.shape[0])))

set_seed(args['seed'])

smiles_to_graph_fuc = None
if args['backbone'] == 'AFP' or args['backbone'] == 'GIN':
    smiles_to_graph_fuc = smiles_2_afpdgl
elif args['backbone'] == 'Weave' or args['backbone'] == 'MPNN':
    smiles_to_graph_fuc = smiles_2_Weave_dgl
else:
    smiles_to_graph_fuc = smiles_2_basedgl

test_set = MolDataset(test_df, smiles_to_graph=smiles_to_graph_fuc)
testloader = DataLoader(test_set, batch_size=args['batch_size'],
                        shuffle=False, drop_last=False,
                        collate_fn=collate_molgraphs)

kf = KFold(n_splits=args['nfold'], shuffle=True)

pair_path = f'data/{dataset}/scaffold/{split}/cluster_res/Hier_{dataset}-{cluster_type}_n={n_cluster}.csv'

if dataset == 'SIDER':
    cluster_res = pd.read_csv(
        f'./data/{dataset}/scaffold/{split}/cluster_res/Hier_{dataset}-{cluster_type}_n={n_cluster}.csv',
        index_col=None, header=None, names=['SMILES', 'cluster_id'])  # index_col=0, header=None, names=['cluster_id'])
    cluster_res['Y'] = pd.read_csv(f'./data/{dataset}/scaffold/{split}/train.csv')[args['label']]
else:
    cluster_res = pd.read_csv(
        f'./data/{dataset}/scaffold/{split}/cluster_res/Hier_{dataset}-{cluster_type}_n={n_cluster}.csv',
        index_col=None, header=None, names=['SMILES', 'cluster_id', 'Y'])

pos_dict, neg_dict = dict(), dict()
for i in range(n_cluster):
    temp = cluster_res[cluster_res['cluster_id'] == i]
    pos_dict[i] = temp[temp['Y'] == 1].index.tolist()
    neg_dict[i] = temp[temp['Y'] == 0].index.tolist()

val_auc_list, val_aupr_list = [], []
test_auc_list, test_aupr_list = [], []

for train_index, test_index in kf.split(all_df):
    train_set = MolDataset(all_df.loc[train_index], smiles_to_graph=smiles_to_graph_fuc)
    valid_set = MolDataset(all_df.loc[test_index], smiles_to_graph=smiles_to_graph_fuc)
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
    esize = train_set[0][2].edata['he'].shape[1] if (args['backbone'] == 'AFP' or
                                                     args['backbone'] == 'Weave' or
                                                     args['backbone'] == 'MPNN') else None
    opt, criterion, model = model_setup(nsize, esize)
    best_val, final_test_res = run(train_loader=train_dataloader,
                                   valid_loader=valid_dataloader,
                                   optimizer=opt,
                                   criterion=criterion,
                                   model=model,
                                   pos_dict=pos_dict, neg_dict=neg_dict,
                                   alpha=args['alpha'])
    val_auc_list.append(best_val['AUROC'])
    val_aupr_list.append(best_val['AUPRC'])
    test_auc_list.append(final_test_res['AUROC'])
    test_aupr_list.append(final_test_res['AUPRC'])

if not os.path.exists(f'res/{dataset}'):
    os.mkdir(f'res/{dataset}')

if not os.path.exists(f'res/{dataset}/summary0223.csv'):
    with open(f'res/{dataset}/summary0223.csv', 'w') as f:
        f.write(f'Model,label,alpha,n_cluster,'
                f'val_AUROC,val_std_AUROC, val_AUPRC, val_std_AUPRC,'
                f'test_AUROC,test_std_AUROC, test_AUPRC, test_std_AUPRC\n')
with open(f'res/{dataset}/summary0223.csv', 'a+') as f:
    f.write(
        f'{args["backbone"]},{args["label"]},{args["alpha"]},{args["n_cluster"]},'
        f'{np.mean(val_auc_list)},{np.std(val_auc_list)},{np.mean(val_aupr_list)},{np.std(val_aupr_list)},'
        f'{np.mean(test_auc_list)},{np.std(test_auc_list)},{np.mean(test_aupr_list)},{np.std(test_aupr_list)}\n')
