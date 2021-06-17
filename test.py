import base_model
import auen_model
import arnn_model

base_model.get_model('base_model', 300)
models = ['auen', 'auen_molecule_CNN_CNN', 'auen_protein_CNN_CNN', 'auen_interaction_CNN_CNN']
auen_model.molecule_model_CNN_CNN(models[1], 32, 8)
auen_model.protein_model_CNN_CNN(models[2], 32, 4)
auen_model.interaction_model(models[3])
models = ['auen', 'auen_molecule_CNN_DNN', 'auen_protein_CNN_DNN', 'auen_interaction_CNN_DNN']
auen_model.molecule_model_CNN_DNN(models[1], 32, 8)
auen_model.protein_model_CNN_DNN(models[2], 32, 4)
models = ['arnn', 'arnn_molecule_RNN_DNN', 'arnn_protein_RNN_DNN', 'arnn_interaction_RNN_DNN']
arnn_model.molecule_model_RNN_DNN(models[1])
arnn_model.protein_model_RNN_DNN(models[2])