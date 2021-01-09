from rdkit import Chem
from rdkit.Chem import Draw

def printMolGraphs(mols):
    chemMols = [(mol.id, Chem.MolFromSmiles(mol.smiles)) for mol in mols]
    return mols_to_pngs(chemMols)

def mols_to_pngs(mols):
    """Helper to write RDKit mols to png files."""
    folderName = 'static/graphs/'
    for mol in mols:
        try:
            filename = "%s%s.png" % (folderName, mol[0])
            Draw.MolToFile(mol[1], filename, (1920,1080))
        except:
            print(f'WARNING: {mol[1]}.png not made')
    return 201
