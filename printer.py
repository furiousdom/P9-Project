from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from IPython.display import Image, display

def display_images(filenames):
    """Helper to pretty-print images."""
    for file in filenames:
        display(Image(file))

def mols_to_pngs(mols, basename="test"):
    """Helper to write RDKit mols to png files."""
    filenames = []
    for i, mol in enumerate(mols):
        filename = "%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename, (1920,1080))
        filenames.append(filename)
    return filenames


def printSequencesToFile(items):
    # Print the sequences
    print(len(items))
    f = open('drugbank_FASTA.txt', 'w', encoding="utf-8")
    for key in items:
        f.write(f'{items[key]}\n')
    f.close()

    # Print the Ids
    f = open('drugbank_IDs.txt', 'w', encoding="utf-8")
    for key in items:
        f.write(f'{key}\n')
    f.close()

def printLabels():
    f = open('drugbank_Labels.txt', 'w', encoding="utf-8")
    for i in range(232):
        if i < 116:
            f.write(f'1\n')
        else:
            f.write(f'0\n')
    f.close()
