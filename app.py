import experiments

prompt1 = 'q'
while prompt1 != '':
    prompt1 = input('Please specify whether you want to "train" or "test" all networks: ')
    if prompt1 == 'train':
        print('Running training....')
        experiments.run_train_session('base_model', 'aau20000', 8, 12)
        experiments.run_train_session('dcnn_model', 'aau20000', 8, 8)
        break
    elif prompt1 == 'test':
        print('Running testing....')
        experiments.run_test_session()
        break
    prompt1 = ''
    print('Wrong input.')
print('Program finished.')

