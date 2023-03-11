import os

datasets = ['BACE']
splits = [7]  # 5, 11, 13, 17
n_clusters = [40]  # [2, 5, 10, 20, 30, 40]
cluster_types = ['ward']  # 'complete'
backbones = ['GAT','GCN','Weave','AFP']
alphas = [0, .01, .02, .03, .04, .05, .1,]

for backbone in backbones:
    for dataset in datasets:
        for n_cluster in n_clusters:
            for cluster_type in cluster_types:
                for sp in splits:
                    for alpha in alphas:
                        os.system(
                            f'sudo /data/rzy/miniconda3/envs/torch17_biomip/bin/python main_GNN.py \
                            --backbone {backbone} \
                            --dataset {dataset} \
                            --split {sp} \
                            --cluster_type {cluster_type} \
                            --n_cluster {n_cluster} \
                            --label Y \
                            --lr {0.01 if (backbone == "GCN" or backbone == "GAT") else 0.001} \
                            --alpha {alpha} \
                            --gpu 0\
                            --batch_size 512')
