import pubchempy as pcp
import pandas as pd
import data_handler
import numpy as np
import requests as r
from Bio import SeqIO
from io import StringIO
from pyfaidx import Fasta
import protvec

LIMIT = 20000

def extract_id(header):
    return header.split('|')[1]

def make_negative_proteins_csv():
    negative_samples_df = pd.read_csv("./data/pub_chem_negative_samples.csv")
    sequences = Fasta('./data/uniprot_sprot.fasta', key_function=extract_id)
    uniprots = negative_samples_df['uniport_accession'].tolist()[:LIMIT]
    fasta_list = []
    protein_problematic_indicies = []
    for i, uniprot in enumerate(uniprots):
        try:
            fasta_list.append(sequences[uniprot])
        except:
            protein_problematic_indicies.append(i)
    data_handler.save_json_obj_to_file('./data/negative_protein_problematic_indicies.json', protein_problematic_indicies)
    # data_handler.save_json_obj_to_file('./data/negative_smiles.json', smiles)

    with open ('./data/negative_proteins.csv', "a") as aau_protein_file:
        for fasta in fasta_list:
            protvec.sequences2protvecsCSV(aau_protein_file, [str(fasta)])

def make_positive_proteins_csv():
    positive_smiles_fasta_pairs = data_handler.load_positive_smiles_fasta_pairs()[:LIMIT]
    positive_fasta_list = [pair[1] for pair in positive_smiles_fasta_pairs]
    indicies = data_handler.load_json_obj_from_file('./data/positive_molecule_problematic_indicies.json')
    for idx in indicies:
        positive_fasta_list.pop(idx)
    with open ('./data/positive_proteins.csv', "a") as aau_protein_file:
        for fasta in positive_fasta_list:
            protvec.sequences2protvecsCSV(aau_protein_file, [str(fasta)])

make_positive_proteins_csv()
# make_negative_proteins_csv()