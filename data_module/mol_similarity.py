import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from tqdm import tqdm
import os


def tanimoto_coefficient(p_vec, q_vec):
    """
    This method implements the cosine tanimoto coefficient metric
    :param p_vec: vector one
    :param q_vec: vector two
    :return: the tanimoto coefficient between vector one and two
    """
    pq = np.dot(p_vec, q_vec)
    # print(pq)
    p_square = np.dot(p_vec, p_vec)
    q_square = np.dot(q_vec, q_vec) # np.linalg.norm(q_vec)
    # print(p_square, q_square)
    return pq / (p_square + q_square - pq)


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))


# calculate morgan fingerprints
def gen_fp(_path, save_path):
    df_smi = pd.read_csv(f"ccks_train.csv")
    smiles = df_smi["smiles"]

    mgf_feat_list = []
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])
        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)
    mgf_feat = np.array(mgf_feat_list, dtype="int64")
    print("morgan feature shape: ", mgf_feat.shape)
    print("saving feature in %s" % save_path)
    np.save(save_path, mgf_feat)


# print(tanimoto_coefficient([1,0,1,0,0,0], [1,0,1,0,0,0]))
mgf_feat = np.load('../../data/raw/train_fp1/mgf_feat.npy').astype(np.int8)
# print(mgf_feat)
sim_mat = []
for m1 in range(mgf_feat.shape[0]):
    sim_row = []
    for m2 in range(mgf_feat.shape[0]):
        sim_row.append(tanimoto_coefficient(mgf_feat[m1], mgf_feat[m2]))
    sim_mat.append(sim_row)
print(sim_mat)
pd.DataFrame(sim_mat).to_csv('sim_mat_train.csv', index=False, header=None)

