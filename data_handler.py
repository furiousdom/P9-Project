import json
import psycopg2
import numpy as np
import pandas as pd
import deepchem as dc
from xdparser import parse_ext_ids, parse_smiles_and_fasta

# ==============================================================================
# General helper functions
# ==============================================================================

def load_json_obj_from_file(file_name):
    json_file = open(file_name, 'r', encoding='utf-8')
    json_obj = json.load(json_file)
    json_file.close()
    return json_obj

def save_json_obj_from_file(file_name, obj):
    json_file = open(file_name, 'w', encoding='utf-8')
    data = json.dumps(obj)
    json_file.write(data)
    json_file.close()

def save_items_to_txt_by_line(file_name, items):
    with open(file_name, 'w', encoding="utf-8") as file:
        for item in items:
            file.write(f'{item}\n')

def save_labels_to_txt_file(file_name, num_of_samples, num_of_positive_samples):
    with open(file_name, 'w', encoding='utf-8') as label_file:
        for i in range(num_of_samples):
            if i < num_of_positive_samples:
                label_file.write('1\n')
            else:
                label_file.write('0\n')

def extract_external_ids(data_frame):
    cids, uniprots = parse_ext_ids('fd')
    ids = []

    for indx, row in data_frame.iterrows():
        cid = str(row['chemical_cid'])
        uniprot = row['uniport_accession']

        if cid in cids.values() and uniprot in uniprots.values():
            temp_cid = list(cids.keys())[list(cids.values()).index(cid)]
            temp_uni = list(uniprots.keys())[list(uniprots.values()).index(uniprot)]
            ids.append((temp_cid, temp_uni))
    return ids

def extract_smiles_list_from_pairs(pairs):
    return [pair[0] for pair in pairs]

def extract_fasta_list_from_pairs(pairs):
    return [pair[1] for pair in pairs]

def save_fasta_for_pyfeat(file_name, smiles_fasta_pairs):
    fasta_list = extract_fasta_list_from_pairs(smiles_fasta_pairs)
    save_items_to_txt_by_line(file_name, fasta_list)

def save_smiles_for_featurizer(file_name, smiles_fasta_pairs):
    smiles_list = extract_smiles_list_from_pairs(smiles_fasta_pairs)
    save_json_obj_from_file(file_name, smiles_list)

def save_molecule_embeddings_to_csv(file_name, molecule_embeddings):
    molecules_data_frame = pd.DataFrame(molecule_embeddings)
    molecules_data_frame.to_csv(file_name)

def id_pairs_to_smiles_fasta_pairs(id_pairs):
    smiles, fasta = parse_smiles_and_fasta('fd')
    pairs = []
    for item in id_pairs:
        try:
            pairs.append((smiles[item[0]], fasta[item[1]]))
        except:
            continue
    return pairs

def connect_db():
    db_session = psycopg2.connect(
        host='localhost',
        port='5432',
        dbname='drugdb',
        user='postgres',
        password='postgres'
    )
    return db_session.cursor()

def print_df_shape_head_tail(df, count):
    print('Shape:\n')
    print(df.shape)
    print()
    print('Head:\n')
    print(df.head(count))
    print()
    print('Tail:\n')
    print(df.tail(count))

# ==============================================================================
# Specialized functions
# ==============================================================================

# ####################################
# Negative Dataset
# ####################################

def extract_negative_external_ids():
    negative_samples_df = pd.read_csv("./data/open_chembl_negative_samples.csv")
    negative_id_pairs = extract_external_ids(negative_samples_df)
    return negative_id_pairs

def save_negative_external_ids():
    negative_id_pairs = extract_negative_external_ids()
    save_json_obj_from_file('./data/negativeExternalIds', negative_id_pairs)

def save_negative_smiles_fasta_pairs(negative_smiles_fasta_pairs):
    save_json_obj_from_file('./data/negativeSmilesFasta.json', negative_smiles_fasta_pairs)

def load_negative_smiles_fasta_pairs():
    negative_smiles_fasta_pairs = load_json_obj_from_file('./data/negativeSmilesFasta.json')
    return negative_smiles_fasta_pairs

def save_negative_fasta_for_pyfeat(negative_smiles_fasta_pairs):
    save_fasta_for_pyfeat('negative_FASTA.txt', negative_smiles_fasta_pairs)

def save_negative_smiles_for_featurizer():
    smiles_fasta_pairs = load_negative_smiles_fasta_pairs()
    save_smiles_for_featurizer('./data/negativeSmiles.json', smiles_fasta_pairs)

def load_negative_smiles_for_featurizer():
    negative_smiles_list = load_json_obj_from_file('./data/negativeSmiles.json')
    return negative_smiles_list

def make_negative_molecule_csv():
    # Don't forget to delete the first row in the newly created file
    featurizer = dc.feat.Mol2VecFingerprint()
    negative_smiles_list = load_negative_smiles_for_featurizer()
    negative_molecules = featurizer(negative_smiles_list)
    save_molecule_embeddings_to_csv('./data/negativeMoleculesDataset.csv', negative_molecules)

# negativeIdPairs = load_json_obj_from_file('./data/negativeIds.json')
# negative_smiles_fasta_pairs = id_pairs_to_smiles_fasta_pairs(negativeIdPairs)
# save_negative_smiles_for_featurizer()
# make_negative_molecule_csv()

# ####################################
# Positive Dataset
# ####################################

def get_positive_id_pairs():
    db_cursor = connect_db()
    db_cursor.execute('SELECT drug_id_1, drug_id_2 FROM public.drug_interactions_table;')
    positive_id_pairs = db_cursor.fetchall()
    return positive_id_pairs

def save_positive_smiles_fasta_pairs(positive_smiles_fasta_pairs):
    save_json_obj_from_file('./data/positive_smiles_fasta.json', positive_smiles_fasta_pairs)

def load_positive_smiles_fasta_pairs():
    positive_smiles_fasta_pairs = load_json_obj_from_file('./data/positive_smiles_fasta.json')
    return positive_smiles_fasta_pairs

def save_positive_fasta_for_pyfeat(positive_smiles_fasta_pairs):
    save_fasta_for_pyfeat('positive_fasta.txt', positive_smiles_fasta_pairs)

def save_positive_smiles_for_featurizer():
    smiles_fasta_pairs = load_positive_smiles_fasta_pairs()
    save_smiles_for_featurizer('./data/positive_smiles_list.json', smiles_fasta_pairs)

def load_positive_smiles_for_featurizer():
    positive_smiles_list = load_json_obj_from_file('./data/positive_smiles_list.json')
    return positive_smiles_list

def make_positive_molecule_csv():
    featurizer = dc.feat.Mol2VecFingerprint()
    positive_smiles_list = load_positive_smiles_for_featurizer()[:20000]
    positive_molecules = featurizer(positive_smiles_list)
    indicies = [i for i, x in enumerate(positive_molecules) if x.size == 0]
    save_json_obj_from_file('./data/problematic_indicies.json', indicies)
    positive_molecules = np.delete(positive_molecules, indicies, 0)
    fixed_positive_molecules = []
    for pos_mol in positive_molecules:
        fixed_positive_molecules.append(list(pos_mol))
    save_molecule_embeddings_to_csv('./data/positive_molecules_dataset.csv', fixed_positive_molecules)

# positive_id_pairs = get_positive_id_pairs()
# positive_smiles_fasta_pairs = id_pairs_to_smiles_fasta_pairs(positive_id_pairs)
# make_positive_molecule_csv()

# ####################################
# Positive & Negative Dataset
# ####################################

def save_proteins_for_featurizer():
    positive_smiles_fasta_pairs = load_positive_smiles_fasta_pairs()[:20000]
    positive_fasta_list = [pair[1] for pair in positive_smiles_fasta_pairs]
    indicies = load_json_obj_from_file('./data/problematic_indicies.json')
    for idx in indicies:
        positive_fasta_list.pop(idx)
    negative_smiles_fasta_pairs = load_negative_smiles_fasta_pairs()
    negative_fasta_list = [pair[1] for pair in negative_smiles_fasta_pairs]
    fasta_list = positive_fasta_list + negative_fasta_list
    save_items_to_txt_by_line('./data/proteins_fasta.txt', fasta_list)

def save_molecule_dataframe_to_csv():
    positive_smiles_list = load_positive_smiles_for_featurizer()[:20000]
    negative_smiles_list = load_negative_smiles_for_featurizer()
    smiles_list = positive_smiles_list + negative_smiles_list
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = featurizer(smiles_list)
    save_molecule_embeddings_to_csv('./data/molecules_dataset.csv', molecules)

def combine_pos_neg_csvs():
    positive_molecules_df = pd.read_csv('./data/positive_molecules_dataset.csv')
    negative_molecules_df = pd.read_csv('./data/negative_molecules_dataset.csv')
    molecules_df = pd.concat([positive_molecules_df, negative_molecules_df])
    molecules_df.to_csv('./data/molecules_dataset.csv')

def read_fastas_from_file(file_name):
    '''
    :param file_name:
    :return: genome sequences
    '''
    with open(file_name, 'r') as file:
        sequences = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                sequences.append(genome.upper())
                genome = ''
        sequences.append(genome.upper())
        del sequences[0]
        return sequences

def make_aau_scores_file(dataset_name, total_number, negative_count):
    with open(f'./data/datasets/{dataset_name}/binding_affinities.txt', 'w') as file:
        for i in range(total_number):
            if i < total_number - negative_count:
                file.write('1\n')
            else:
                file.write('0\n')

make_aau_scores_file('aau20000', 20027, 241)