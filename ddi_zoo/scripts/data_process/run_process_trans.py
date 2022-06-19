import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
import pdb


def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''
    return ' '.join(tokens)

def clean_smiles(smiles):
    t = re.sub(':\d*', '', smiles)
    return t

def main():
    print("processing start !")

    raw_data_dir = "/home/v-jiaclin/blob2/v-xialiang/dmp/data/new_split/transductive/"

    inductive_data_dir = "/home/v-jiaclin/blob2/v-xialiang/dmp/data/new_split/inductive/new_build3/"
    raw_train_dir = "{}{}".format(raw_data_dir, 'drugbank_id_smiles.csv')

    all_drug_dict = {}
    drugbank_csv = pd.read_csv(raw_train_dir)
    for idx, smiles in zip(tqdm(drugbank_csv['DrugID']), drugbank_csv['Smiles']):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        all_drug_dict[idx] = smi_tokenizer(smiles)
    
    train_csv_file = "{}{}".format(inductive_data_dir, 'train.csv')
    valid_csv_file = "{}{}".format(inductive_data_dir, 'valid.csv')
    test_csv_file = "{}{}".format(inductive_data_dir, 'test.csv')
    
    train_a_dir = "{}{}".format(inductive_data_dir, 'train.a')
    train_b_dir = "{}{}".format(inductive_data_dir, 'train.b')
    valid_a_dir = "{}{}".format(inductive_data_dir, 'valid.a')
    valid_b_dir = "{}{}".format(inductive_data_dir, 'valid.b')
    test_a_dir = "{}{}".format(inductive_data_dir, 'test.a')
    test_b_dir = "{}{}".format(inductive_data_dir, 'test.b')

    train_nega_dir = "{}{}".format(inductive_data_dir, 'train.nega')
    train_negb_dir = "{}{}".format(inductive_data_dir, 'train.negb')
    valid_nega_dir = "{}{}".format(inductive_data_dir, 'valid.nega')
    valid_negb_dir = "{}{}".format(inductive_data_dir, 'valid.negb')
    test_nega_dir = "{}{}".format(inductive_data_dir, 'test.nega')
    test_negb_dir = "{}{}".format(inductive_data_dir, 'test.negb')

    train_label_dir = "{}{}".format(inductive_data_dir, 'train.label')
    valid_label_dir = "{}{}".format(inductive_data_dir, 'valid.label')
    test_label_dir = "{}{}".format(inductive_data_dir, 'test.label')
    
    label_dict = {}
    with open(train_a_dir, 'w') as ta_w, open(train_nega_dir, 'w') as tna_w, \
        open(train_b_dir, 'w') as tb_w, open(train_negb_dir, 'w') as tnb_w, \
        open(valid_a_dir, 'w') as va_w, open(valid_nega_dir, 'w') as vna_w, \
        open(valid_b_dir, 'w') as vb_w, open(valid_negb_dir, 'w') as vnb_w, \
        open(test_a_dir, 'w') as tsa_w, open(test_nega_dir, 'w') as tsna_w, \
        open(test_b_dir, 'w') as tsb_w, open(test_negb_dir, 'w') as tsnb_w, \
        open(train_label_dir, 'w') as tl, open(valid_label_dir, 'w') as vl, \
        open(test_label_dir, 'w') as tsl:
        
        train_csv = pd.read_csv(train_csv_file)
        for a, b, na, nb, y in zip(tqdm(train_csv['id1']), train_csv['id2'], \
            train_csv['neg_id1'], train_csv['neg_id2'], train_csv['y']):
            
            # ta_w.writelines(all_drug_dict[a] + '\n')
            # tb_w.writelines(all_drug_dict[b] + '\n')

            # tna_w.writelines(all_drug_dict[na] + '\n')
            # tnb_w.writelines(all_drug_dict[nb] + '\n')

            # pdb.set_trace()
            # tl.writelines(str(y) + '\n')
            if str(y) not in label_dict:
                label_dict[str(y)] = len(label_dict)
    
        valid_csv = pd.read_csv(valid_csv_file)
        for a, b, na, nb, y in zip(tqdm(valid_csv['id1']), valid_csv['id2'], \
            valid_csv['neg_id1'], valid_csv['neg_id2'], valid_csv['y']):
            
            # va_w.writelines(all_drug_dict[a] + '\n')
            # vb_w.writelines(all_drug_dict[b] + '\n')

            # vna_w.writelines(all_drug_dict[na] + '\n')
            # vnb_w.writelines(all_drug_dict[nb] + '\n')

            # vl.writelines(str(y) + '\n')
            if str(y) not in label_dict:
                label_dict[str(y)] = len(label_dict)

        # test_csv = pd.read_csv(test_csv_file)
        # for a, b, na, nb, y in zip(tqdm(test_csv['id1']), test_csv['id2'], \
        #     test_csv['neg_id1'], test_csv['neg_id2'], test_csv['y']):
            
        #     # tsa_w.writelines(all_drug_dict[a] + '\n')
        #     # tsb_w.writelines(all_drug_dict[b] + '\n')

        #     # tsna_w.writelines(all_drug_dict[na] + '\n')
        #     # tsnb_w.writelines(all_drug_dict[nb] + '\n')

        #     # tsl.writelines(str(y) + '\n')
        #     if str(y) not in label_dict:
        #         label_dict[str(y)] = len(label_dict)

    pdb.set_trace()
    label_dict_dir  = "{}{}".format(inductive_data_dir, 'label.dict')
    # with open(label_dict_dir, 'w') as label_dict_w:
    #     for label_name, label_idx in label_dict.items():
    #         label_dict_w.writelines(label_name + " " + str(label_idx) + '\n')

    print("processing done !")
    

if __name__ == "__main__":
    main()
