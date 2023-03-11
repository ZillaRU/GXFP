import ssl

from sklearn.cluster import AgglomerativeClustering, KMeans

ssl._create_default_https_context = ssl._create_unverified_context
import os
import numpy as np
import pandas as pd
import sklearn.cluster as sc


def cluster_mols(dpath, Xn, method, fname, n_cluster=8):
    def rec_cluster_res(note=""):
        dir = os.path.join(dpath,'cluster_res')
        if not os.path.exists(dir):
            os.mkdir(dir)
        if fname != 'SIDER':
            with open(os.path.join(dir, method + '_' + fname + note + '.csv'), 'w') as f:
                gt = pd.read_csv(os.path.join(dpath,f"train.csv"))['Y']
                for i in range(len(pred_y)):
                    f.write(f'{Xn.index.values[i]},{pred_y[i]},{gt[i]}\n')
            return
        # no label
        with open(os.path.join(dir, method + '_' + fname + note + '.csv'), 'w') as f:
            for i in range(len(pred_y)):
                f.write(f'{Xn.index.values[i]},{pred_y[i]}\n')

    if method == 'MeanShift':
        bw = sc.estimate_bandwidth(Xn, n_samples=len(Xn), quantile=0.01)
        model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
        # model.fit(Xn)  # 完成聚类
        pred_y = model.fit_predict(Xn)  # 预测点在哪个聚类中
        rec_cluster_res()
    elif method == 'Hier':
        for linkage in ('ward', 'complete'):  # 'average'
            model = AgglomerativeClustering(linkage=linkage, n_clusters=n_cluster)
            pred_y = model.fit_predict(Xn)  # 预测点在哪个聚类中
            rec_cluster_res(note='-' + linkage + "_n=" + str(n_cluster))
    elif method == 'Kmeans':
        model = KMeans(n_clusters=n_cluster, init='k-means++', n_init=20, random_state=28)
        pred_y = model.fit_predict(Xn)  # 预测点在哪个聚类中
        rec_cluster_res(note="n=" + str(n_cluster))


import sys

if __name__ == "__main__":
    dataset = sys.argv[1]
    dpath = f'./data/{dataset}/scaffold'
    for sp in [28,7,17]:
        X = np.load(os.path.join(dpath, str(sp), "mgf_feat.npy"))
        id_X = pd.DataFrame(X,
                            index=pd.read_csv(f'{dpath}/{sp}/train.csv')['SMILES'],
                            columns=[i for i in range(X.shape[1])])
        # cluster_mols(train_X, method='MeanShift', fname='train_only')
        # cluster_mols(eval_X, method='MeanShift', fname='valid_only')
        # cluster_mols(pd.concat([train_X,eval_X],axis=0), method='MeanShift', fname='train+valid')

        # cluster_mols(train_X, method='Hier', fname='train_only')
        # cluster_mols(eval_X, method='Hier', fname='valid_only')
        # cluster_mols(pd.concat([train_X, eval_X], axis=0), method='Hier', fname='train+valid')
        # # Hier average complete n=8 效果差
        if len(sys.argv) == 2:
            # cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=2)
            cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=5)
            cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=10)
            cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=20)
            # cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=30)
            cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=40)
        else:
            cluster_mols(f'{dpath}/{sp}', id_X, method='Hier', fname=dataset, n_cluster=int(sys.argv[2]))
        # cluster_mols(dpath, id_X, method='Hier', fname=dataset, n_cluster=5)
        # cluster_mols(dpath, id_X, method='Hier', fname=dataset, n_cluster=10)
        # cluster_mols(dpath, id_X, method='Hier', fname=dataset, n_cluster=20)
        # cluster_mols(dpath, id_X, method='Hier', fname=dataset, n_cluster=30)
        # cluster_mols(dpath, id_X, method='Hier', fname=dataset, n_cluster=40)
        # cluster_mols(train_X, method='Kmeans', fname='train_only')
        # cluster_mols(pd.concat([train_X, eval_X], axis=0), method='Kmeans', fname='train+valid')
        #
        # cluster_mols(train_X, method='Kmeans', fname='train_only', n_cluster=30)
        # cluster_mols(eval_X, method='Kmeans', fname='valid_only', n_cluster=30)
        # cluster_mols(pd.concat([train_X, eval_X], axis=0), method='Kmeans', fname='train+valid', n_cluster=30)
