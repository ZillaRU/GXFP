import sys

import deepchem as dc
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    dataset = sys.argv[1]
    _dir = f"{dataset}/scaffold"
    all_smiles = pd.read_csv(f"{dataset}/{dataset}.csv")['SMILES']
    Xs = np.zeros(len(all_smiles))
    Ys = np.ones(len(all_smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=Ys, w=np.zeros(len(all_smiles)), ids=all_smiles)
    scaffoldsplitter = dc.splits.ScaffoldSplitter()

    _df = pd.read_csv(f"{sys.argv[1]}/{sys.argv[1]}.csv")
    for seed in [28,17,7]:
        os.mkdir(os.path.join(_dir, f"{seed}"))
        train, valid, test = scaffoldsplitter.split(dataset)
        _df.iloc[train].to_csv(os.path.join(_dir, f"{seed}/train.csv"), index=False)
        _df.iloc[valid].to_csv(os.path.join(_dir, f"{seed}/valid.csv"), index=False)
        _df.iloc[test].to_csv(os.path.join(_dir, f"{seed}/test.csv"), index=False)
    # _dir = f"{dataset}/split"
    # all_data = pd.read_csv(f"{dataset}/{dataset}.csv")
    # if os.path.exists(_dir):
    #     os.system(f'rm -rf {_dir}')
    # os.mkdir(_dir)d
    # for seed in [28]:
    #     os.mkdir(os.path.join(_dir, f"{seed}"))
    #     train_data, test_data = train_test_split(all_data, train_size=0.2, test_size=0.8, random_state=seed)
    #     train_data.to_csv(os.path.join(_dir, f"{seed}/train.csv"), index=False)
    #     test_data.to_csv(os.path.join(_dir, f"{seed}/test.csv"), index=False)
