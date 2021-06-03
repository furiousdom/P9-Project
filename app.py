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
########## DON'T UNCOMMENT BASE MODEL ##########

print('Start Training')
models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
models = ['arnn', 'arnn_molecule_RNN_RNN', 'arnn_protein_RNN_RNN', 'arnn_interaction_RNN_RNN']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
models = ['auen', 'auen_molecule_CNN_DNN', 'auen_protein_CNN_DNN', 'auen_interaction_CNN_DNN']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
models = ['arnn', 'arnn_molecule_RNN_DNN', 'arnn_protein_RNN_DNN', 'arnn_interaction_RNN_DNN']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
print('Finished Training')
