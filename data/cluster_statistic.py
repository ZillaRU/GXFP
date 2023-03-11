import matplotlib.pyplot as plt
import pandas as pd


for dataset in ['_HIV', 'BACE']:
    for sp in [5,7,11,13,17]:
        for n in [5,10,20,40]:
            for type in ['complete', 'ward']:
                df = pd.read_csv(f'{dataset}/split/{sp}/cluster_res/Hier_{dataset}-{type}_n={n}.csv', header=None,
                                 names=['SMILES','cluster','Y'])
                cnts = df.value_counts('cluster',sort=False)
                # cnts = df.value_counts(['cluster','Y'],sort=False)
                print(cnts)
                # with open()
# men_means = [20, 35, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# width = 0.35
# plt.bar(labels, men_means, width)
#
# min_std = 0.
#
# # 关键在bottom参数
# plt.bar(labels, women_means, width, bottom=men_means)
# plt.title('Stacked bar')
# plt.show()
