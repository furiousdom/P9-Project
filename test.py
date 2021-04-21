from pubchempy import Compound
import pandas as pd
import data_handler
import numpy as np
import deepchem as dc

LIMIT = 20000

def make_positive_molecule_csv(smiles_list):
    featurizer = dc.feat.Mol2VecFingerprint()
    print('Starting featurizing')
    positive_molecules = featurizer(smiles_list)
    print('Ended featurizing')
    indicies = [i for i, x in enumerate(positive_molecules) if x.size == 0]
    data_handler.save_json_obj_to_file('./data/negative_molecule_problematic_indicies.json', indicies)
    positive_molecules = np.delete(positive_molecules, indicies, 0)
    fixed_positive_molecules = []
    for pos_mol in positive_molecules:
        fixed_positive_molecules.append(list(pos_mol))
    data_handler.save_molecule_embeddings_to_csv('./data/negative_molecules.csv', fixed_positive_molecules)

negative_samples_df = pd.read_csv("./data/pub_chem_negative_samples.csv")
cids = negative_samples_df['chemical_cid'].tolist()[:LIMIT]
smiles_list = []
for cid in cids:
    c = Compound.from_cid(cid).to_dict()
    smiles_list.append(c['canonical_smiles'])
# data_handler.save_json_obj_to_file('./data/negative_smiles.json', smiles)
make_positive_molecule_csv(smiles_list)