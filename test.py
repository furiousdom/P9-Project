from xdparser import parse
from utils import save_json_obj_to_file
from utils import read_fastas_from_file
from utils import load_json_obj_from_file

molecules = parse('fd', 'SMILES')
molecules = list(molecules.values())
print(f'Length of molecules: {len(molecules)}')
save_json_obj_to_file('./data/drugbank-molecules.json', molecules)

proteins = read_fastas_from_file('./data/uniprot_sprot.fasta')[:12000]
print(len(proteins))
proteins = set(proteins)
print(f'Length of proteins: {len(proteins)}')
proteins = list(proteins)
save_json_obj_to_file('./data/drugbank-proteins.json', proteins)