import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from utils import set_seed


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--split', type=str, default='28')
    parser.add_argument('--label', type=str, default='Y')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--nfold', type=int, default=5)
    return parser.parse_args().__dict__


def model_setup():
    if args['backbone'] == 'SVM':
        model = svm.SVC(kernel='poly', class_weight='balanced',
                           max_iter=2000,
                           probability=True)
    elif args['backbone'] == 'LR':
        model = LogisticRegression(random_state=0)
    elif args['backbone'] == 'RF':
        model = RandomForestClassifier(min_samples_leaf=2,
                                       n_estimators=1190,
                                       n_jobs=-1,
                                       max_features='log2',
                                       criterion='entropy',
                                       max_depth=50,
                                       class_weight='balanced'
                                       )
    else:
        raise NotImplementedError
    return model


args = get_args()
dataset, split = args['dataset'], args['split']
all_df = pd.read_csv(f'data/{dataset}/scaffold/{split}/train.csv')
all_df.insert(all_df.shape[1], 'id', list(range(all_df.shape[0])))
all_X = pd.DataFrame(np.load(f'data/{dataset}/scaffold/{split}/mgf_feat.npy'))
all_df = pd.concat([all_df, all_X], axis=1)
print(all_X)

test_df = pd.read_csv(f'data/{dataset}/scaffold/{split}/test.csv')
test_df.insert(test_df.shape[1], 'id', list(range(test_df.shape[0])))
test_X = pd.DataFrame(np.load(f'data/{dataset}/scaffold/{split}/test_mgf_feat.npy'))
test_df = pd.concat([test_df, test_X], axis=1)

set_seed(args['seed'])

kf = KFold(n_splits=args['nfold'], shuffle=True)

print(test_X.shape, test_df.shape)
print(all_df)

val_auc_list, val_aupr_list = [], []
test_auc_list, test_aupr_list = [], []

for train_index, test_index in kf.split(all_df):
    train_set = all_df.loc[train_index]
    model = model_setup()
    model.fit(train_set.loc[:, all_df.shape[1]-2048:], train_set[args['label']])
    pred = model.predict_proba(test_df.loc[:, test_df.shape[1]-2048:])[:, 1].tolist()
    val_auc_list.append(0)
    val_aupr_list.append(0)
    test_auc_list.append(roc_auc_score(test_df[args['label']].tolist(), pd.Series(pred)))
    test_aupr_list.append(average_precision_score(test_df[args['label']].tolist(), pd.Series(pred)))

if not os.path.exists(f'res/{dataset}'):
    os.mkdir(f'res/{dataset}')

if not os.path.exists(f'res/{dataset}/summary0223.csv'):
    with open(f'res/{dataset}/summary0223.csv', 'w') as f:
        f.write(f'Model,label,alpha,n_cluster,'
                f'val_AUROC,val_std_AUROC, val_AUPRC, val_std_AUPRC,'
                f'test_AUROC,test_std_AUROC, test_AUPRC, test_std_AUPRC\n')
with open(f'res/{dataset}/summary0223.csv', 'a+') as f:
    f.write(
        f'{args["backbone"]},{args["label"]},-,-,'
        f'{np.mean(val_auc_list)},{np.std(val_auc_list)},{np.mean(val_aupr_list)},{np.std(val_aupr_list)},'
        f'{np.mean(test_auc_list)},{np.std(test_auc_list)},{np.mean(test_aupr_list)},{np.std(test_aupr_list)}\n')
