import protvec
import data_handler
import numpy as np
import pandas as pd
import requests as r
import deepchem as dc
import pubchempy as pcp
from Bio import SeqIO
from io import StringIO
from pyfaidx import Fasta
from pubchempy import Compound

LIMIT = 20000

####################################### Start of test.py ###########

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

####################################### End of test.py ###########

def extract_id(header):
    return header.split('|')[1]

def make_negative_proteins_csv():
    negative_samples_df = pd.read_csv("./data/pub_chem_negative_samples.csv")
    sequences = Fasta('./data/uniprot_sprot.fasta', key_function=extract_id)
    uniprots = negative_samples_df['uniport_accession'].tolist()[:LIMIT]
    fasta_list = []
    protein_problematic_indicies = []
    for i, uniprot in enumerate(uniprots):
        if i % 10 == 0:
            print(i)
        try:
            fasta_list.append(str(sequences[uniprot]))
        except:
            protein_problematic_indicies.append(i)
    data_handler.save_json_obj_to_file('./data/negative_protein_problematic_indicies.json', protein_problematic_indicies)
    data_handler.save_json_obj_to_file('./data/negative_fastas.json', fasta_list)

def featurize_from_json():
    fasta_list = data_handler.load_json_obj_from_file('./data/negative_fastas.json')
    counter = 0
    with open ('./data/negative_proteins.csv', "a") as file:
        for fasta in fasta_list:
            if counter % 100 == 0:
                print(counter)
            counter += 1
            protvec.sequences2protvecsCSV(file, [str(fasta)])

    # fasta_list = load_json_obj_from_file('./data/negative_fastas.json')
    # indicies = data_handler.load_json_obj_from_file('./data/negative_molecule_problematic_indicies.json')
    # for idx in reversed(indicies):
    #     fasta_list.pop(idx)
    # with open ('./data/negative_proteins.csv', "a") as aau_protein_file:
    #     for fasta in fasta_list:
    #         protvec.sequences2protvecsCSV(aau_protein_file, [str(fasta)])

def make_positive_proteins_csv():
    positive_smiles_fasta_pairs = data_handler.load_positive_smiles_fasta_pairs()[:LIMIT]
    positive_fasta_list = [pair[1] for pair in positive_smiles_fasta_pairs]
    indicies = data_handler.load_json_obj_from_file('./data/positive_molecule_problematic_indicies.json')
    counter = 0
    for idx in reversed(indicies):
        positive_fasta_list.pop(idx)
    with open ('./data/positive_proteins.csv', "a") as aau_protein_file:
        for fasta in positive_fasta_list:
            if counter % 100 == 0:
                print(counter)
            counter += 1
            protvec.sequences2protvecsCSV(aau_protein_file, [str(fasta)])

# make_positive_proteins_csv()  ####RUN this later tonight.
# make_negative_proteins_csv()
# featurize_from_json()

# data_handler.combine_pos_neg_molecules_csvs()
# print('Done with combining molecules')
# data_handler.combine_pos_neg_proteins_csvs()
# print('Done with combining proteins')
# data_handler.make_aau_scores_file('aau40000', 39451, 19664)


molecules = pd.read_csv("./data/datasets/aau40000/molecules.csv")
print(molecules.shape)
# molecules.drop(index = proteins, inplace=True)
molecules.drop(molecules.filter(regex="Unname"),axis=1, inplace=True)
print(molecules.shape)
# print(negative_molecules)
# print(negative_proteins)
# print(len(negative_proteins))
data_handler.save_molecule_embeddings_to_csv('./data/datasets/aau40000/molecules.csv', molecules)
