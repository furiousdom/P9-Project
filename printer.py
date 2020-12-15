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