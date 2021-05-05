import protvec
import data_handler
import numpy as np
import deepchem as dc
from data_handler import load_json_obj_from_file
from data_handler import save_items_to_txt_by_line
from data_handler import save_molecule_embeddings_to_csv

start = 30056
limit = 70000

############################################################################
# Kiba dataset
############################################################################

def featurize_kiba():
    kiba_no_features = []
    kiba_json = load_json_obj_from_file('./data/kiba.json')[start:limit]
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = []
    proteins = []

    for i, pair in enumerate(kiba_json):
        try:
            molecules.append(featurizer(pair[1])[0])
        except:
            print(f'Molecule {i} was not appended.')
            kiba_no_features.append(i)

    molecules = np.delete(molecules, kiba_no_features, 0)
    data_handler.save_molecule_embeddings_to_csv('./data/kiba_molecules_rest2.csv', molecules)
    del molecules

    with open ('./data/kiba_proteins_rest2.csv', "a") as kiba_protein_file:
        for i, pair in enumerate(kiba_json):
            if i not in kiba_no_features:
                protvec.sequences2protvecsCSV(kiba_protein_file, [pair[0]])

    kiba_scores = []

    for i in range(start - start, limit - start):
        if i + start not in kiba_no_features:
            kiba_scores.append(kiba_json[i][2])

    f = open('./data/kiba_scores_rest2.txt', 'w')
    for i in range(start - start, limit - start):
        f.write(str(kiba_scores[i]) + '\n')
    f.close()

featurize_kiba()

# ############################################################################
# # Any dataset
# ############################################################################

def molecule_protein_positions(dataset_name):
    return 1, 0 if dataset_name == 'kiba' else 0, 1

def save_molecule_feature_vectors(dataset_name, molecules, problematic_indicies):
    mol_embed_file_name = './data/' + dataset_name + '_molecules.csv'
    save_molecule_embeddings_to_csv(mol_embed_file_name, molecules)
    indicies_file_name = './data/' + dataset_name + 'molecule_problematic_indicies.txt'
    save_items_to_txt_by_line(indicies_file_name, problematic_indicies)

def featurize_molecules(dataset_name, json_dataset, molecule_idx):
    featurizer = dc.feat.Mol2VecFingerprint()
    problematic_indicies = []
    molecules = []
    for i, pair in enumerate(json_dataset):
        try:
            molecules.append(featurizer(pair[molecule_idx])[0])
        except:
            print(f'Molecule {i} was not appended.')
            problematic_indicies.append(i)
    molecules = np.delete(molecules, problematic_indicies, 0)
    save_molecule_feature_vectors(dataset_name, molecules, problematic_indicies)
    return problematic_indicies

def featurize_proteins(dataset_name, json_dataset, protein_idx, mol_problematic_indxs):
    with open ('./data/' + dataset_name + '_proteins.csv', "a") as protein_file:
        for i, pair in enumerate(json_dataset):
            if i not in mol_problematic_indxs:
                protvec.sequences2protvecsCSV(protein_file, [pair[protein_idx]])

def save_binding_affinities(dataset_name, json_dataset, problematic_indicies):
    scores_file_path = './data/' + dataset_name + 'binding_affinities.txt'
    with open(scores_file_path, 'w') as scores_file:
        for i in range(0, limit - start):
            if i + start not in problematic_indicies:
                scores_file.write(str(json_dataset[i][2]) + '\n')

def featurize_dataset(dataset_name):
    dataset_path = './data/' + dataset_name + '.json'
    json_dataset = load_json_obj_from_file(dataset_path)[start:limit]
    molecule_idx, protein_idx = molecule_protein_positions(dataset_name)
    mol_problematic_indxs = featurize_molecules(dataset_name, json_dataset, molecule_idx)
    featurize_proteins(dataset_name, json_dataset, protein_idx, mol_problematic_indxs)
    save_binding_affinities(dataset_name, json_dataset, mol_problematic_indxs)
