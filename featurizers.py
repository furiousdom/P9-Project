import numpy as np
import deepchem as dc
import libs.protvec as protvec
from utils import load_json_obj_from_file
from utils import save_items_to_txt_by_line
from utils import save_molecule_embeddings_to_csv

start = 100000 # 70000
limit = 118254 # 100000

def molecule_protein_positions(dataset_name):
    return (1, 0) if dataset_name == 'kiba' else (0, 1)

def save_molecule_feature_vectors(dataset_name, molecules, problematic_indicies):
    mol_embed_file_name = './data/datasets/' + dataset_name + '/rest_molecules.csv'
    save_molecule_embeddings_to_csv(mol_embed_file_name, molecules)
    indicies_file_name = './data/datasets/' + dataset_name + '/rest_molecule_problematic_indicies.txt'
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
    with open ('./data/datasets/' + dataset_name + '/rest_proteins.csv', "a") as protein_file:
        for i, pair in enumerate(json_dataset):
            if i not in mol_problematic_indxs:
                protvec.sequences2protvecsCSV(protein_file, [pair[protein_idx]])

def save_binding_affinities(dataset_name, json_dataset, problematic_indicies):
    scores_file_path = './data/datasets/' + dataset_name + '/rest_binding_affinities.txt'
    with open(scores_file_path, 'w') as scores_file:
        for i in range(0, limit - start):
            if i + start not in problematic_indicies:
                scores_file.write(str(json_dataset[i][2]) + '\n')

def featurize_dataset(dataset_name):
    dataset_path = './data/' + dataset_name + '.json'
    json_dataset = load_json_obj_from_file(dataset_path)[start:]
    molecule_idx, protein_idx = molecule_protein_positions(dataset_name)
    mol_problematic_indxs = featurize_molecules(dataset_name, json_dataset, molecule_idx)
    featurize_proteins(dataset_name, json_dataset, protein_idx, mol_problematic_indxs)
    save_binding_affinities(dataset_name, json_dataset, mol_problematic_indxs)

featurize_dataset('kiba')
