import os
import pandas as pd
import re
from rdkit import Chem
from tqdm import tqdm
import numpy as np


raw_dir = '/home/v-xialiang/blob2_containter/v-xialiang/dmp/data/new_split/transductive'

# inductive file
trans_smiles = "{}/{}".format(raw_dir, "drugbank_id_smiles.csv")
trans_train = "{}/{}".format(raw_dir, 'pair_pos_neg_triples_train_fold2.csv')
trans_test = "{}/{}".format(raw_dir, 'pair_pos_neg_triples_test_fold2.csv')

def check_new_build():
    
    reb_dir = "{}/{}".format(raw_dir, "new_build")
    new_trans_train = "{}/{}".format(reb_dir, 'train.csv')
    new_trans_valid = "{}/{}".format(reb_dir, 'valid.csv')
    new_trans_test = "{}/{}".format(reb_dir, 'test.csv')

    drug_pair = set()
    for file in [new_trans_train, new_trans_valid, new_trans_test]:
        file_csv = pd.read_csv(file)
        for d1, d2 in zip(tqdm(file_csv['id1']), file_csv['id2']):
            drug_pair.add((d1, d2))
            drug_pair.add((d2, d1))

    print("drug dict size = {}".format(len(drug_pair)))

    for file in [new_trans_train, new_trans_valid, new_trans_test]:
        file_csv = pd.read_csv(file)
        for nd1, nd2 in zip(tqdm(file_csv['neg_id1']), file_csv['neg_id2']):
            assert (nd1, nd2) not in drug_pair


def rebuild_drug_pair():

    all_drug_pair = set()
    drug_dict = {}
    for file in [trans_train, trans_test]:
        file_csv = pd.read_csv(file)
        for d1, d2 in zip(tqdm(file_csv['Drug1_ID']), file_csv['Drug2_ID']):
            all_drug_pair.add((d1, d2))
    
            if d1 not in drug_dict:
                drug_dict[d1] = set([d2])
            else:
                drug_dict[d1].add(d2)
            if d2 not in drug_dict:
                drug_dict[d2] = set([d1])
            else:
                drug_dict[d2].add(d1)

    print("all drug pair len = {}".format(len(all_drug_pair)))
    print("drug_dict len = {}".format(len(drug_dict)))

    # limit the negative sample from training set
    all_drug_set = drug_dict.keys()

    reb_dir = "{}/{}".format(raw_dir, "new_build2")
    if not os.path.exists(reb_dir):
        os.makedirs(reb_dir)

    new_trans_train = "{}/{}".format(reb_dir, 'train.csv')
    new_trans_valid = "{}/{}".format(reb_dir, 'valid.csv')
    new_trans_test = "{}/{}".format(reb_dir, 'test.csv')
    
    train_data = []
    valid_data = []
    train_csv = pd.read_csv(trans_train)
    for d1, d2, y in zip(tqdm(train_csv['Drug1_ID']), train_csv['Drug2_ID'], train_csv['Y']):

        if np.random.rand() > 0.5:
            neg_d1 = d1
            candidate_list = list(all_drug_set - drug_dict[d1])
            candidate_idx = np.random.choice(len(candidate_list), 1)[0]
            neg_d2 = candidate_list[candidate_idx]
        else:
            candidate_list = list(all_drug_set - drug_dict[d2])
            candidate_idx = np.random.choice(len(candidate_list), 1)[0]
            neg_d1 = candidate_list[candidate_idx]
            neg_d2 = d2
        
        data_dict = {
                'id1': d1,
                'id2': d2,
                'neg_id1': neg_d1,
                'neg_id2': neg_d2,
                'y': y,
            }

        if np.random.rand() < 0.25:
            valid_data.append(data_dict)
        else:
            train_data.append(data_dict)
    df_data = pd.DataFrame(train_data)
    df_data.to_csv(new_trans_train)
    df_data = pd.DataFrame(valid_data)
    df_data.to_csv(new_trans_valid)

    test_data = []
    test_csv = pd.read_csv(trans_test)
    for d1, d2, y in zip(tqdm(test_csv['Drug1_ID']), test_csv['Drug2_ID'], test_csv['Y']):

        if np.random.rand() > 0.5:
            neg_d1 = d1
            candidate_list = list(all_drug_set - drug_dict[d1])
            candidate_idx = np.random.choice(len(candidate_list), 1)[0]
            neg_d2 = candidate_list[candidate_idx]
        else:
            candidate_list = list(all_drug_set - drug_dict[d2])
            candidate_idx = np.random.choice(len(candidate_list), 1)[0]
            neg_d1 = candidate_list[candidate_idx]
            neg_d2 = d2
        
        data_dict = {
                'id1': d1,
                'id2': d2,
                'neg_id1': neg_d1,
                'neg_id2': neg_d2,
                'y': y,
            }
        test_data.append(data_dict)
    
    df_data = pd.DataFrame(test_data)
    df_data.to_csv(new_trans_test)


def main():
    
    # check_new_build()

    rebuild_drug_pair()

if __name__ == "__main__":
    
    main()
