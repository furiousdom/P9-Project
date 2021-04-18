from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from IPython.display import Image, display

def display_images(file_names):
    """Helper to pretty-print images."""
    for file in file_names:
        display(Image(file))

def mols_to_pngs(mols, basename="test"):
    """Helper to write RDKit mols to png files."""
    file_names = []
    for i, mol in enumerate(mols):
        file_name = "%s%d.png" % (basename, i)
        Draw.MolToFile(mol, file_name, (1920,1080))
        file_names.append(file_name)
    return file_names


def print_sequences_to_file(items):
    sequences_file = open('drugbank_FASTA.txt', 'w', encoding='utf-8')
    ids_file = open('drugbank_IDs.txt', 'w', encoding='utf-8')
    for key in items:
        sequences_file.write(f'{items[key]}\n')
        ids_file.write(f'{key}\n')
    sequences_file.close()
    ids_file.close()

def print_labels_to_file():
    f = open('drugbank_Labels.txt', 'w', encoding='utf-8')
    for i in range(232):
        if i < 116:
            f.write(f'1\n')
        else:
            f.write(f'0\n')
    f.close()
