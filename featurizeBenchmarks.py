import dataHandler
import deepchem as dc
import protvec

limit = 10000
watcher = 0

############################################################################
# Kiba dataset
############################################################################

# kibaNofeatures = []
# kibaJson = dataHandler.loadJsonObjFromFile('./data/kiba.json')[:limit]
# featurizer = dc.feat.Mol2VecFingerprint()
# molecules = []
# proteins = []

# for i, pair in enumerate(kibaJson):
#     watcher += 1
#     print(watcher)
#     try:
#         molecules.append(featurizer(pair[1])[0])
#     except:
#         kibaNofeatures.append(i)

# dataHandler.saveMoleculeEmbeddingsToCsv('./data/kibaMolecules.csv', molecules)
# del molecules

# for i, pair in enumerate(kibaJson):
#     watcher += 1
#     print(watcher)
#     if i not in kibaNofeatures:
#         proteins.append(pair[0])

# protvec.sequences2protvecsCSV('./data/kibaProteins.csv', proteins)

# ############################################################################
# # Davis dataset
# ############################################################################

# davisNofeatures = []
# featurizer = dc.feat.Mol2VecFingerprint()
# molecules = []
# proteins = []

# davisDataset = []
# with open('./data/davis.txt') as file:
#     for i, line in enumerate(file):
#         if i < limit:
#             davisDataset.append(line.split(' '))

# for i, pair in enumerate(davisDataset):
#     watcher += 1
#     print(watcher)
#     try:
#         molecules.append(featurizer(pair[0])[0])
#     except:
#         davisNofeatures.append(i)

# dataHandler.saveMoleculeEmbeddingsToCsv('./data/davisMolecules.csv', molecules)
# del molecules

# for i, pair in enumerate(davisDataset):
#     watcher += 1
#     print(watcher)
#     if i not in davisNofeatures:
#         proteins.append(pair[0])

# protvec.sequences2protvecsCSV('./data/davisProteins.csv', proteins)

############################################################################
# Scores
############################################################################

def loadDatasetFromTxt():
    davisDataset = []
    with open('./data/davis.txt') as file:
        for i, line in enumerate(file):
            if i < limit:
                davisDataset.append(line.split(' '))
    return davisDataset

kibaJson = dataHandler.loadJsonObjFromFile('./data/kiba.json')[:limit]
davisDataset = loadDatasetFromTxt()

davisScores = []
kibaScores = []

for i in range(limit):
    davisScores.append(davisDataset[i][2])
    kibaScores.append(kibaJson[i][2])

f = open('./data/davisScores.txt', 'w')
for i in range(limit):
    f.write(str(davisScores[i]))
f.close()

f = open('./data/kibaScores.txt', 'w')
for i in range(limit):
    f.write(str(kibaScores[i]) + '\n')

f.close()

