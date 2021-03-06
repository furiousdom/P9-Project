import json
import psycopg2
# import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from xdparser import parseExtIds, parseSMILESandFASTA

# ==============================================================================
# General helper functions
# ==============================================================================

def loadJsonObjFromFile(fileName):
    jsonFile = open(fileName, 'r', encoding='utf-8')
    jsonObj = json.load(jsonFile)
    jsonFile.close()
    return jsonObj

def saveJsonObjToFile(fileName, obj):
    jsonFile = open(fileName, 'w', encoding='utf-8')
    data = json.dumps(obj)
    jsonFile.write(data)
    jsonFile.close()

def saveItemsToTxt(fileName, items):
    print(len(items))
    f = open(fileName, 'w', encoding="utf-8")
    for item in items:
        f.write(f'{item}\n')
    f.close()

def extractExternalIds(dataFrame):
    cids, uniprots = parseExtIds('fd')
    ids = []

    for indx, row in dataFrame.iterrows():
        cid = str(row['chemical_cid'])
        uniprot = row['uniport_accession']

        if cid in cids.values() and uniprot in uniprots.values():
            tempcid = list(cids.keys())[list(cids.values()).index(cid)]
            tempuni = list(uniprots.keys())[list(uniprots.values()).index(uniprot)]
            ids.append((tempcid, tempuni))
    return ids

def extractSmilesListFromPairs(pairs):
    return [pair[0] for pair in pairs]

def extractFastaListFromPairs(pairs):
    return [pair[1] for pair in pairs]

def saveFastaForPyFeat(fileName, smilesFastaPairs):
    fastaList = extractFastaListFromPairs(smilesFastaPairs)
    saveItemsToTxt(fileName, fastaList)

def saveSmilesForFeaturizer(fileName, pairs):
    smilesList = extractSmilesListFromPairs(pairs)
    saveJsonObjToFile(fileName, smilesList)

def saveMoleculeEmbeddingsToCsv(fileName, moleculeEmbeddings):
    moleculesDataFrame = pd.DataFrame(moleculeEmbeddings)
    moleculesDataFrame.to_csv(fileName)

def idPairsToSmilesFastaPairs(idPairs):
    smiles, fasta = parseSMILESandFASTA('fd')
    pairs = []

    for item in idPairs:
        try:
            pairs.append((smiles[item[0]], fasta[item[1]]))
        except:
            continue
    return pairs

def connectDb():
    dbSession = psycopg2.connect(
        host='localhost',
        port='5432',
        dbname='drugdb',
        user='postgres',
        password='postgres'
    )
    return dbSession.cursor()

# ==============================================================================
# Specialized functions
# ==============================================================================

# ####################################
# Negative Dataset
# ####################################

def extractNegativeExternalIds():
    negativeSamplesDf = pd.read_csv("./data/openChemblNegativeSamples.csv")
    negativeIdPairs = extractExternalIds(negativeSamplesDf)
    return negativeIdPairs

def saveNegativeExternalIds():
    negativeIdPairs = extractNegativeExternalIds()
    saveJsonObjToFile('./data/negativeExternalIds', negativeIdPairs)

def saveNegativeSmilesFastaPairs(negativeSmilesFastaPairs):
    saveJsonObjToFile('./data/negativeSmilesFasta.json', negativeSmilesFastaPairs)

def loadNegativeSmilesFastaPairs():
    negativeSmilesFastaPairs = loadJsonObjFromFile('./data/negativeSmilesFasta.json')
    return negativeSmilesFastaPairs

def saveNegativeFastaForPyFeat(negativeSmilesFastaPairs):
    saveFastaForPyFeat('negative_FASTA.txt', negativeSmilesFastaPairs)

def saveNegativeSmilesForFeaturizer():
    smilesFastaPairs = loadNegativeSmilesFastaPairs()
    saveSmilesForFeaturizer('./data/negativeSmiles.json', smilesFastaPairs)

def loadNegativeSmilesForFeaturizer():
    negativeSmilesList = loadJsonObjFromFile('./data/negativeSmilesList.json')
    return negativeSmilesList

def makeNegativeMoleculeCsv():
    # Don't forget to delete the first row in the newly created file
    featurizer = dc.feat.Mol2VecFingerprint()
    negativeSmilesList = loadNegativeSmilesForFeaturizer()
    negativeMolecules = featurizer(negativeSmilesList)
    saveMoleculeEmbeddingsToCsv('./data/negativeMoleculesDataset.csv', negativeMolecules)

# negativeIdPairs = loadJsonObjFromFile('./data/negativeIds.json')
# negativeSmilesFastaPairs = idPairsToSmilesFastaPairs(negativeIdPairs)

# ####################################
# Positive Dataset
# ####################################

def getPositiveIdPairs():
    dbCursor = connectDb()
    dbCursor.execute('SELECT drug_id_1, drug_id_2 FROM public.drug_interactions_table;')
    positiveIdPairs = dbCursor.fetchall()
    return positiveIdPairs

def savePositiveSmilesFastaPairs(positiveSmilesFastaPairs):
    saveJsonObjToFile('./data/positiveSmilesFasta.json', positiveSmilesFastaPairs)

def loadPositiveSmilesFastaPairs():
    positiveSmilesFastaPairs = loadJsonObjFromFile('./data/positiveSmilesFasta.json')
    return positiveSmilesFastaPairs

def savePositiveFastaForPyfeat(positiveSmilesFastaPairs):
    saveFastaForPyFeat('positive_FASTA.txt', positiveSmilesFastaPairs)

def savePositiveSmilesForFeaturizer():
    smilesFastaPairs = loadPositiveSmilesFastaPairs()
    saveSmilesForFeaturizer('./data/positiveSmilesList.json', smilesFastaPairs)

def loadPositiveSmilesForFeaturizer():
    positiveSmilesList = loadJsonObjFromFile('./data/positiveSmilesList.json')
    return positiveSmilesList

def makePositiveMoleculeCsv():
    # Don't forget to delete the first row in the newly created file
    featurizer = dc.feat.Mol2VecFingerprint()
    positiveSmilesList = loadPositiveSmilesForFeaturizer()
    positiveMolecules = featurizer(positiveSmilesList)
    saveMoleculeEmbeddingsToCsv('./data/positiveMoleculesDataset.csv', positiveMolecules)

# positiveIdPairs = getPositiveIdPairs()
# positiveSmilesFastaPairs = idPairsToSmilesFastaPairs(positiveIdPairs)

# ==============================================================================
# Program
# ==============================================================================

def printShapeAndHead(df):
    print('Shape')
    print(df.shape)
    print()
    print('Head')
    print(df.head(4))

negativeMoleculesDataFrame = pd.read_csv('./data/negativeMoleculesDataset.csv')
negativeProteinsDataFrame = pd.read_csv('./data/negativeProteinsFullDataset.csv')

negativeDataFrame = pd.concat([negativeProteinsDataFrame, negativeMoleculesDataFrame], axis=1)

neg_x = np.array(negativeDataFrame)
neg_y = np.zeros((negativeDataFrame.shape[0],), dtype=int)

print(neg_x.shape)
print(neg_y.shape)

x_train, x_test, y_train, y_test = train_test_split(neg_x, neg_y, test_size=0.2, random_state=0)

# positiveMoleculesDataFrame = pd.read_csv('./data/positiveMoleculesDataset.csv', nrows=3000)
# positiveProteinsDataFrame = pd.read_csv('./data/positiveProteinsFullDataset.csv', nrows=3000)
# print('Positive Molecules CSV')
# printShapeAndHead(positiveMoleculesDataFrame)
# print('Positive Proteins CSV')
# printShapeAndHead(positiveProteinsDataFrame)

# positiveDataFrame = pd.concat([positiveProteinsDataFrame, positiveMoleculesDataFrame], axis=1)
# print('Combined dataset')
# printShapeAndHead(positiveDataFrame)

# # df = pd.concat([negativeDataFrame, positiveDataFrame])

model = tf.keras.models.Sequential()

model.add(layers.Input(shape=(x_train.shape[1],)))
model.add(layers.Dense(471, activation="relu"))
model.add(layers.Dense(300, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(25, activation="relu"))
model.add(layers.Dense(1, activation="relu"))

model.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.MeanSquaredError()
)

model.fit(x_train, y_train, batch_size = 1, epochs = 1)

predictions = model.predict(x_test)
