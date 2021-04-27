from pubchempy import Compound
import pandas as pd
import data_handler
import numpy as np
import deepchem as dc

LIMIT = 20000

# negative_samples_df = pd.read_csv("./data/pub_chem_negative_samples.csv")
# cids = negative_samples_df['chemical_cid'].tolist()[:LIMIT]
# print('Starting lookup')
# smiles_list = []
# counter = 0
# for cid in cids:
#     c = Compound.from_cid(cid).to_dict()
#     smiles_list.append(c['canonical_smiles'])
#     if counter % 10 == 0:
#         print(counter)
#     counter += 1
# data_handler.save_json_obj_to_file('./data/negative_smiles.json', smiles_list)
# print('Ended lookup')


# smiles_list = data_handler.load_json_obj_from_file('./data/negative_smiles.json')
# featurizer = dc.feat.Mol2VecFingerprint()
# print('Starting featurizing')
# positive_molecules = featurizer(smiles_list)
# print('Ended featurizing')
# indicies = [i for i, x in enumerate(positive_molecules) if x.size == 0]
# data_handler.save_json_obj_to_file('./data/negative_molecule_problematic_indicies.json', indicies)
# positive_molecules = np.delete(positive_molecules, indicies, 0)
# fixed_positive_molecules = []
# for pos_mol in positive_molecules:
#     fixed_positive_molecules.append(list(pos_mol))

negative_molecules = pd.read_csv("./data/negative_molecules.csv")
negative_proteins = data_handler.load_json_obj_from_file('./data/negative_protein_problematic_indicies.json')
print(negative_molecules.shape)
negative_molecules.drop(index = negative_proteins, inplace=True)
negative_molecules.drop(negative_molecules.filter(regex="Unname"),axis=1, inplace=True)
print(negative_molecules.shape)
# print(negative_molecules)
# print(negative_proteins)
print(len(negative_proteins))
data_handler.save_molecule_embeddings_to_csv('./data/negative_molecules.csv', negative_molecules)


