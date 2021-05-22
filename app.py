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


experiments.run_network_train_session('base_model', 1, 'kiba', epochs=128, batch_size=256)
experiments.run_network_train_session('base_model', 2, 'davis', epochs=128, batch_size=256)
experiments.run_network_train_session('dcnn_model', 1, 'kiba', epochs=100, batch_size=256)
experiments.run_network_train_session('dcnn_model', 2, 'davis', epochs=100, batch_size=256)
models = ['auen', 'auen_molecule', 'auen_protein', 'auen_interaction']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
models = ['arnn', 'arnn_molecule', 'arnn_protein', 'arnn_interaction']
experiments.run_autoencoder_train_session(models, 1, 'kiba', epochs=100, batch_size=256)
experiments.run_autoencoder_train_session(models, 2, 'davis', epochs=100, batch_size=256)
