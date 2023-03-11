import os
import sys

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from tqdm import tqdm


def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))


def getTmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True))


def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]


def gen_fp(source_path, save_path):
    df_smi = pd.read_csv(source_path)
    smiles = df_smi["SMILES"]
    mgf_feat_list = []
    maccs_feat_list = []
    t_mg_feat_list = []
    for ii in tqdm(range(len(smiles))):
        # print(smiles.iloc[ii])
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])
        print(smiles.iloc[ii])
        mgf = getmorganfingerprint(rdkit_mol)
        print(smiles.iloc[ii])
        mgf_feat_list.append(mgf)

        maccs = getmaccsfingerprint(rdkit_mol)
        maccs_feat_list.append(maccs)

        t_mg = getTmorganfingerprint(rdkit_mol)
        t_mg_feat_list.append(t_mg)

    mgf_feat = np.array(mgf_feat_list, dtype="int64")
    # maccs_feat = np.array(maccs_feat_list, dtype="int64")
    # t_mg_feat = np.array(t_mg_feat_list, dtype="int64")
    print("morgan feature shape: ", mgf_feat.shape)
    # print("maccs feature shape: ", maccs_feat.shape)
    # print("T morgan feature shape: ", t_mg_feat.shape)
    print("saving feature in %s" % save_path)
    _ = save_path[save_path.rfind('/')+1:save_path.rfind('.')]
    np.save(os.path.join(save_path, f"test_mgf_feat.npy"), mgf_feat)
    # np.save(os.path.join(save_path, f"{_}_maccs_feat.npy"), maccs_feat)
    # np.save(os.path.join(save_path, f"{_}_t_mg_feat.npy"), t_mg_feat)


# python calc_fp.py DATASET
if __name__ == "__main__":
    ds_name = sys.argv[1]
    for sp in [28]:
        gen_fp(os.path.join('./data', ds_name, 'scaffold', str(sp), 'test.csv'),
               os.path.join('./data', ds_name, 'scaffold', str(sp)))
        # gen_fp(os.path.join('./data', ds_name, 'scaffold', str(sp), 'train.csv'),
        #        os.path.join('./data', ds_name, 'scaffold', str(sp)))
