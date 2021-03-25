import data_handler
import deepchem as dc
import protvec

limit = 10000

############################################################################
# Kiba dataset
############################################################################

kibaNofeatures = []
kibaJson = data_handler.loadJsonObjFromFile('./data/kiba.json')[:limit]
featurizer = dc.feat.Mol2VecFingerprint()
molecules = []
proteins = []

for i, pair in enumerate(kibaJson):
    try:
        molecules.append(featurizer(pair[1])[0])
    except:
        kibaNofeatures.append(i)

# # dataHandler.saveMoleculeEmbeddingsToCsv('./data/kibaMolecules.csv', molecules)
# # del molecules

with open ('./data/kibaProteins.csv', "a") as kiba_protein_file:
    for i, pair in enumerate(kibaJson):
        if i not in kibaNofeatures:
            protvec.sequences2protvecsCSV(kiba_protein_file, [pair[0]])

# ############################################################################
# # Davis dataset
# ############################################################################

davisNofeatures = []
featurizer = dc.feat.Mol2VecFingerprint()
molecules = []
proteins = []

davisDataset = []
with open('./data/davis.txt') as file:
    for i, line in enumerate(file):
        if i < limit:
            davisDataset.append(line.split(' '))

for i, pair in enumerate(davisDataset):
    try:
        molecules.append(featurizer(pair[0])[0])
    except:
        print(f'Molecule {i} was not appended.')
        davisNofeatures.append(i)

# dataHandler.saveMoleculeEmbeddingsToCsv('./data/davisMolecules.csv', molecules)
# del molecules

with open ('./data/davisProteins.csv', "a") as davis_protein_file:
    for i, pair in enumerate(davisDataset):
        if i not in davisNofeatures:
            protvec.sequences2protvecsCSV(davis_protein_file, [pair[1]])

############################################################################
# Scores
############################################################################

# def loadDatasetFromTxt():
#     davisDataset = []
#     with open('./data/davis.txt') as file:
#         for i, line in enumerate(file):
#             if i < limit:
#                 davisDataset.append(line.split(' '))
#     return davisDataset

# kibaJson = dataHandler.loadJsonObjFromFile('./data/kiba.json')[:limit]
# davisDataset = loadDatasetFromTxt()

# davisScores = []
# kibaScores = []

# for i in range(limit):
#     davisScores.append(davisDataset[i][2])
#     kibaScores.append(kibaJson[i][2])

# f = open('./data/davisScores.txt', 'w')
# for i in range(limit):
#     f.write(str(davisScores[i]))
# f.close()

# f = open('./data/kibaScores.txt', 'w')
# for i in range(limit):
#     f.write(str(kibaScores[i]) + '\n')

# f.close()

