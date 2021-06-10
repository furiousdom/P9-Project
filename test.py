from utils import load_json_obj_from_file
from utils import read_fastas_from_file

smiles_list = load_json_obj_from_file('./data/positive_smiles_list.json')

smiles_set = set(smiles_list)

print(f'length of smiles list: {len(smiles_list)}')
print(f'length of smiles set: {len(smiles_set)}')

fasta_list = read_fastas_from_file('./data/proteins_fasta.txt')

fasta_set = set(fasta_list)

print(f'length of fasta list: {len(fasta_list)}')
print(f'length of fasta set: {len(fasta_set)}')

davis_dataset = load_json_obj_from_file('./data/davis.json')

davis_smiles_list = [pair[0] for pair in davis_dataset]
davis_fasta_list = load_json_obj_from_file('./data/davis_proteins.txt')

davis_smiles_set = set(davis_smiles_list)

print(f'length of davis_smiles_list: {len(davis_smiles_list)}')
print(f'length of davis_smiles_set: {len(davis_smiles_set)}')

davis_fasta_set = set(davis_fasta_list)

print(f'length of davis_fasta_list: {len(davis_fasta_list)}')
print(f'length of davis_fasta_set: {len(davis_fasta_set)}')

kiba_dataset = load_json_obj_from_file('./data/kiba.json')

kiba_smiles_list = [pair[1] for pair in kiba_dataset]
kiba_fasta_list = [pair[0] for pair in kiba_dataset]

kiba_smiles_set = set(kiba_smiles_list)

print(f'length of kiba_smiles_list: {len(kiba_smiles_list)}')
print(f'length of kiba_smiles_set: {len(kiba_smiles_set)}')

kiba_fasta_set = set(kiba_fasta_list)

print(f'length of kiba_fasta_list: {len(kiba_fasta_list)}')
print(f'length of kiba_fasta_set: {len(kiba_fasta_set)}')

kiba_davis_smiles = kiba_smiles_set.union(davis_smiles_set)
print(f'length of kiba_davis_smiles: {len(kiba_davis_smiles)}')

complete_smiles_set = kiba_davis_smiles.union(smiles_set)
print(f'length of complete_smiles_set: {len(complete_smiles_set)}')

kiba_davis_fasta = kiba_fasta_set.union(davis_fasta_set)
print(f'length of kiba_davis_fasta: {len(kiba_davis_fasta)}')

complete_fasta_set = kiba_davis_fasta.union(fasta_set)
print(f'length of complete_fasta_set: {len(complete_fasta_set)}')
