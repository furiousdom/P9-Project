import psycopg2
import pandas as pd
import deepchem as dc
from xdparser import parse, parseExtIds, readProteinIds
import json

# ==============================================================================
# Don't know what this section does.
# ==============================================================================

# class Molecule:
#     def __init__(self, id=None, embedding=None):
#         self.id = id
#         self.embedding = embedding

# proteinEmbeddings = pd.read_csv("./data/optimumDataset.csv")
# proteinIds = readProteinIds()
# molecules = parse('fd', 'SMILES')

# proteins = {}
# for indx, row in proteinEmbeddings.iterrows():
#     proteins[proteinIds[indx]] = row

# featurizer = dc.feat.Mol2VecFingerprint()
# moleculeEmbeddings = featurizer(molecules.values())

# print(f'Molcules Embeddings length: {len(moleculeEmbeddings)}')
# print(f'Molcules length: {len(molecules)}')

# pairs = []
# for i in range(len(molecules)):
#     pairs.append(Molecule(molecules.keys[i], moleculeEmbeddings[i]))

# ==============================================================================
# Extracting id's for negative dataset.
# ==============================================================================

cids, uniprots = parseExtIds('fd')
negativeSamples = pd.read_csv("./data/negative_samples.csv")

negativeDataset = []

for indx, row in negativeSamples.iterrows():
    cid = str(row['chemical_cid'])
    uniprot = row['uniport_accession']

    if cid in cids.values() and uniprot in uniprots.values():
        print('hella')
        tempcid = list(cids.keys())[list(cids.values()).index(cid)]
        tempuni = list(uniprots.keys())[list(uniprots.values()).index(uniprot)]
        negativeDataset.append((tempcid, tempuni))

def saveToJson(data):
    jsonData = json.dumps(data)
    f = open('negativeDataset.json', 'w', encoding='utf-8')
    f.write(jsonData)
    f.close()


# def printNegativeSamples(items):
#     print(len(items))
#     f = open('negative.txt', 'w', encoding="utf-8")
#     for item in items:
#         f.write(f'{item}\n')
#     f.close()

# printNegativeSamples(negativeDataset)

# ==============================================================================
#
# ==============================================================================

# dbSession = psycopg2.connect(
#     host='localhost',
#     port='5432',
#     dbname='drugdb',
#     user='postgres',
#     password='postgres'
# )

# dbCursor = dbSession.cursor()

# molecules = parse('fd', 'SMILES')
