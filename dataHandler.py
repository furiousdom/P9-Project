import json
import psycopg2
import pandas as pd
import deepchem as dc
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

def saveLabelsTxtFile(fileName, num_of_samples, num_of_positive_samples):
    f = open(fileName, 'w', encoding='utf-8')
    for i in range(num_of_samples):
        if i < num_of_positive_samples:
            f.write('1\n')
        else:
            f.write('0\n')

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

def printDfShapeHeadTail(df, count):
    print('Shape:\n')
    print(df.shape)
    print()
    print('Head:\n')
    print(df.head(count))
    print()
    print('Tail:\n')
    print(df.tail(count))

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
    # If errors arive while featurizing the molecules, it will screw up the CSV file
    # Loading 1000 samples doesn't yield errors meaning CSV is created normally
    # Don't forget to delete the first row in the newly created file
    featurizer = dc.feat.Mol2VecFingerprint()
    positiveSmilesList = loadPositiveSmilesForFeaturizer()
    positiveMolecules = featurizer(positiveSmilesList)
    saveMoleculeEmbeddingsToCsv('./data/positiveMoleculesDataset.csv', positiveMolecules)

# positiveIdPairs = getPositiveIdPairs()
# positiveSmilesFastaPairs = idPairsToSmilesFastaPairs(positiveIdPairs)

# ####################################
# Positive & Negative Dataset
# ####################################

def saveProteinsForFeaturizer():
    positiveSmilesFastaPairs = loadPositiveSmilesFastaPairs()[:1000]
    positiveFastaList = [pair[1] for pair in positiveSmilesFastaPairs]
    negativeSmilesFastaPairs = loadNegativeSmilesFastaPairs()
    negativeFastaList = [pair[1] for pair in negativeSmilesFastaPairs]
    fastaList = positiveFastaList + negativeFastaList
    saveItemsToTxt('./data/proteins_FASTA.txt', fastaList)
    # Change numbers in line below as needed
    saveLabelsTxtFile('./data/labels_FASTA.txt', 1241, 1000)

def saveMoleculeDataFrameToCsv():
    positiveSmilesList = loadPositiveSmilesForFeaturizer()[:1000]
    negativeSmilesList = loadNegativeSmilesForFeaturizer()
    smilesList = positiveSmilesList + negativeSmilesList
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = featurizer(smilesList)
    saveMoleculeEmbeddingsToCsv('./data/moleculeDataset.csv', molecules)

def readFASTAsFromFile(fileName):

    '''
    :param fileName:
    :return: genome sequences
    '''
    with open(fileName, 'r') as file:
        sequences = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                sequences.append(genome.upper())
                genome = ''
        sequences.append(genome.upper())
        del sequences[0]
        return sequences
