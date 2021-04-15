import math
import json
import psycopg2
import pandas as pd
import deepchem as dc
from xdparser import parseExtIds, parseSMILESandFASTA
import numpy as np

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
    negativeSmilesList = loadJsonObjFromFile('./data/negativeSmiles.json')
    return negativeSmilesList

def makeNegativeMoleculeCsv():
    # Don't forget to delete the first row in the newly created file
    featurizer = dc.feat.Mol2VecFingerprint()
    negativeSmilesList = loadNegativeSmilesForFeaturizer()
    negativeMolecules = featurizer(negativeSmilesList)
    saveMoleculeEmbeddingsToCsv('./data/negativeMoleculesDataset.csv', negativeMolecules)

# negativeIdPairs = loadJsonObjFromFile('./data/negativeIds.json')
# negativeSmilesFastaPairs = idPairsToSmilesFastaPairs(negativeIdPairs)
# saveNegativeSmilesForFeaturizer()
makeNegativeMoleculeCsv()

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
    positiveSmilesList = loadPositiveSmilesForFeaturizer()[:20000]
    positiveMolecules = featurizer(positiveSmilesList)
    indicies = [i for i, x in enumerate(positiveMolecules) if x.size == 0]
    saveJsonObjToFile('./data/problematic_indicies.json', indicies)
    positiveMolecules = np.delete(positiveMolecules, indicies, 0)
    fixed_positive_molecules = []
    for l in positiveMolecules:
        fixed_positive_molecules.append(list(l))
    saveMoleculeEmbeddingsToCsv('./data/positiveMoleculesDataset.csv', fixed_positive_molecules)

# positiveIdPairs = getPositiveIdPairs()
# positiveSmilesFastaPairs = idPairsToSmilesFastaPairs(positiveIdPairs)
# makePositiveMoleculeCsv()

# ####################################
# Positive & Negative Dataset
# ####################################

def saveProteinsForFeaturizer():
    positiveSmilesFastaPairs = loadPositiveSmilesFastaPairs()[:20000]
    positiveFastaList = [pair[1] for pair in positiveSmilesFastaPairs]
    indicies = loadJsonObjFromFile('./data/problematic_indicies.json')
    for idx in indicies:
        positiveFastaList.pop(idx)
    negativeSmilesFastaPairs = loadNegativeSmilesFastaPairs()
    negativeFastaList = [pair[1] for pair in negativeSmilesFastaPairs]
    fastaList = positiveFastaList + negativeFastaList
    saveItemsToTxt('./data/proteins_FASTA.txt', fastaList)
    # Change numbers in line below as needed
    # saveLabelsTxtFile('./data/labels_FASTA.txt', 1241, 1000)

def saveMoleculeDataFrameToCsv():
    positiveSmilesList = loadPositiveSmilesForFeaturizer()[:1000]
    negativeSmilesList = loadNegativeSmilesForFeaturizer()
    smilesList = positiveSmilesList + negativeSmilesList
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = featurizer(smilesList)
    saveMoleculeEmbeddingsToCsv('./data/moleculeDataset.csv', molecules)

def combine_pos_neg_csvs():
    positive_molecules_df = pd.read_csv('./data/positiveMoleculesDataset.csv')
    negative_molecules_df = pd.read_csv('./data/negativeMoleculesDataset.csv')
    frames = [positive_molecules_df, negative_molecules_df]
    molecules_df = pd.concat(frames)
    molecules_df.to_csv('./data/molecules_dataset.csv')

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

def load_binary_scores(filename, threshold, preprocess = False):
    scores_list = []
    f = open(filename, 'r')
    for line in f:
        score = float(line)
        if preprocess:
            score = -1 * math.log10(score/pow(10, 9))
        if score >= threshold:
            scores_list.append([0, 1])
        else:
            scores_list.append([1, 0])
    f.close()
    return scores_list

df = pd.read_csv('./data/copy_molecules_dataset.csv')
df.drop(index = 0)
# df.drop(index = 1)
df.to_csv('./data/copy2.csv')
print(df)