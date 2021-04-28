import protvec
import data_handler
import deepchem as dc
import numpy as np

start = 10000
limit = 30056

############################################################################
# Kiba dataset
############################################################################

def featurize_kiba():
    kiba_no_features = []
    kiba_json = data_handler.load_json_obj_from_file('./data/kiba.json')[start:limit]
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = []
    proteins = []

    for i, pair in enumerate(kiba_json):
        try:
            molecules.append(featurizer(pair[1])[0])
        except:
            print(f'Molecule {i} was not appended.')
            kiba_no_features.append(i)

    molecules = np.delete(molecules, kiba_no_features, 0)
    data_handler.save_molecule_embeddings_to_csv('./data/kiba_molecules_rest.csv', molecules)
    del molecules

    with open ('./data/kiba_proteins_rest.csv', "a") as kiba_protein_file:
        for i, pair in enumerate(kiba_json):
            if i not in kiba_no_features:
                protvec.sequences2protvecsCSV(kiba_protein_file, [pair[0]])
                
    kiba_scores = []

    for i in range(start - start, limit - start):
        if i + start not in kiba_no_features:
            kiba_scores.append(kiba_json[i][2])

    f = open('./data/kiba_scores_rest.txt', 'w')
    for i in range(start - start, limit - start):
        f.write(str(kiba_scores[i]) + '\n')
    f.close()

# featurize_kiba()

# ############################################################################
# # Davis dataset
# ############################################################################

def featurize_davis():
    davis_no_features = []
    featurizer = dc.feat.Mol2VecFingerprint()
    molecules = []
    proteins = []

    davis_dataset = []
    with open('./data/davis.txt') as file:
        for i, line in enumerate(file):
            if i >= start and i < limit:
                davis_dataset.append(line.split(' '))

    # for i, pair in enumerate(davis_dataset):
    #     try:
    #         molecules.append(featurizer(pair[0])[0])
    #     except:
    #         print(f'Molecule {i} was not appended.')
    #         davis_no_features.append(i)
    
    # molecules = np.delete(molecules, davis_no_features, 0)
    # data_handler.save_molecule_embeddings_to_csv('./data/davis_molecules_rest.csv', molecules)
    # del molecules

    # with open ('./data/davis_proteins_rest.csv', "a") as davis_protein_file:
    #     for i, pair in enumerate(davis_dataset):
    #         if i not in davis_no_features:
    #             protvec.sequences2protvecsCSV(davis_protein_file, [pair[1]])

    davis_scores = []

    for i in range(start - start, limit - start):
        if i + start not in davis_no_features:
            davis_scores.append(davis_dataset[i][2])

    f = open('./data/davis_scores_rest.txt', 'w')
    for i in range(start - start, limit - start):
        f.write(str(davis_scores[i]))
    f.close()

featurize_davis()

############################################################################
# Scores
############################################################################

# def load_dataset_from_txt():
#     davis_dataset = []
#     with open('./data/davis.txt') as file:
#         for i, line in enumerate(file):
#             if i < limit:
#                 davis_dataset.append(line.split(' '))
#     return davis_dataset

# kiba_json = data_handler.load_json_obj_from_file('./data/kiba.json')[:limit]
# davis_dataset = load_dataset_from_txt()

# kibaScores = []

# for i in range(limit):
#     kibaScores.append(kiba_json[i][2])

# f = open('./data/kibaScores.txt', 'w')
# for i in range(limit):
#     f.write(str(kibaScores[i]) + '\n')
# f.close()

# # ############################################################################
# # # AAU dataset
# # ############################################################################

# aau_proteins = data_handler.read_fastas_from_file('./data/protein_sequences.txt')

# with open ('./data/aau_proteins.csv', "a") as aau_protein_file:
#     for protein in aau_proteins:
#         protvec.sequences2protvecsCSV(aau_protein_file, [protein])