# # ##### UNCOMMENT THE TWO LINES BELOW IF YOU WANT TO RUN ON CPU #####
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import experiments

# prompt1 = 'q'
# while prompt1 != '':
#     prompt1 = input('Please specify whether you want to "train" or "test" all networks: ')
#     if prompt1 == 'train':
#         print('Running training....')
#         break
#     elif prompt1 == 'test':
#         print('Testing not implemented')
#         # print('Running testing....')
#         break
#     prompt1 = ''
#     print('Wrong input.')
# print('Program finished.')

########## DON'T UNCOMMENT BASE MODEL ##########
# experiments.run_network_train_session('base_model', 1, 'kiba', epochs=128, batch_size=256)
# experiments.run_network_train_session('base_model', 2, 'davis', epochs=128, batch_size=256)
# experiments.run_network_train_session('dcnn_model', 1, 'kiba', epochs=100, batch_size=256)
# experiments.run_network_train_session('dcnn_model', 2, 'davis', epochs=100, batch_size=256)
# models = ['arnn', 'arnn_molecule_RNN_RNN', 'arnn_protein_RNN_RNN', 'arnn_interaction_RNN_RNN']
# experiments.run_train_session(models, 21, 'kiba', epochs=100, batch_size=256)
# experiments.run_train_session(models, 22, 'davis', epochs=100, batch_size=256)
########## DON'T UNCOMMENT BASE MODEL ##########

# models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
# experiments.run_train_session(models, 25, 'kiba', epochs=100, batch_size=256, load_weights=False)
# experiments.run_train_session(models, 26, 'davis', epochs=100, batch_size=256, load_weights=False)
# models = ['auen', 'auen_molecule_CNN_DNN', 'auen_protein_CNN_DNN', 'auen_interaction_CNN_DNN']
# experiments.run_train_session(models, 25, 'kiba', epochs=100, batch_size=256, load_weights=False)
# experiments.run_train_session(models, 26, 'davis', epochs=100, batch_size=256, load_weights=False)
# models = ['arnn', 'arnn_molecule_RNN_DNN', 'arnn_protein_RNN_DNN', 'arnn_interaction_RNN_DNN']
# experiments.run_train_session(models, 25, 'kiba', epochs=100, batch_size=256, load_weights=False)
# experiments.run_train_session(models, 26, 'davis', epochs=100, batch_size=256, load_weights=False)


########## RUN TESTING OF INTERACTION MODELS WITH FOLDS ##########
# models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
# experiments.run_test_session_with_folds(models, 25, 'kiba')
# experiments.run_test_session_with_folds(models, 26, 'davis')
# models = ['auen', 'auen_molecule_CNN_DNN', 'auen_protein_CNN_DNN', 'auen_interaction_CNN_DNN']
# experiments.run_test_session_with_folds(models, 25, 'kiba')
# experiments.run_test_session_with_folds(models, 26, 'davis')
# models = ['arnn', 'arnn_molecule_RNN_DNN', 'arnn_protein_RNN_DNN', 'arnn_interaction_RNN_DNN']
# experiments.run_test_session_with_folds(models, 25, 'kiba')
# experiments.run_test_session_with_folds(models, 26, 'davis')
########## RUN TESTING OF INTERACTION MODELS WITH FOLDS ##########

########## RUN TRAINING OF INTERACTION MODELS WITH FULL DATASETS ##########
models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
experiments.run_retraining_with_full_datasets(models, 27, 'kiba', epochs=100, batch_size=256)
experiments.run_retraining_with_full_datasets(models, 28, 'davis', epochs=100, batch_size=256)
# models = ['auen', 'auen_molecule_CNN_DNN', 'auen_protein_CNN_DNN', 'auen_interaction_CNN_DNN']
# experiments.run_retraining_with_full_datasets(models, 27, 'kiba', epochs=100, batch_size=256)
# experiments.run_retraining_with_full_datasets(models, 28, 'davis', epochs=100, batch_size=256)
# models = ['arnn', 'arnn_molecule_RNN_DNN', 'arnn_protein_RNN_DNN', 'arnn_interaction_RNN_DNN']
# experiments.run_retraining_with_full_datasets(models, 27, 'kiba', epochs=100, batch_size=256)
# experiments.run_retraining_with_full_datasets(models, 28, 'davis', epochs=100, batch_size=256)
########## RUN TRAINING OF INTERACTION MODELS WITH FULL DATASETS ##########

# BEFORE RUNNING app.py AGAIN
    # Cut and paste the results into Full_dataset_results folder
    # Comment out lines 55-57
    # Un-comment lines 70-72
    # Change train_size back to 0.33 in get_simple_dataset_split function (experiments.py line 31)

########## RUN TRAINING OF MOLECULE AUTOENCODER WITH MORE EPOCHS ##########
########## DON'T FORGET TO CHANGE train_size BACK TO 0.33 IN get_simple_dataset_split ##########
# models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
# experiments.run_training_with_more_epochs(models, 29, 'kiba', epochs=200, batch_size=256)
# experiments.run_training_with_more_epochs(models, 30, 'davis', epochs=200, batch_size=256)
########## RUN TRAINING OF MOLECULE AUTOENCODER WITH MORE EPOCHS ##########
