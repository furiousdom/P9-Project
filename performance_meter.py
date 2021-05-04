import math
import emetrics
import numpy as np
from sklearn.metrics import auc
from data_loader import DATASETS_TO_PREPROCESS
from data_handler import save_predictions
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import average_precision_score as sk_ap
# from lifelines.utils import concordance_index

def calc_accuracy(y_test, predictions):
    counter = 0
    for i in range(y_test.shape[0]):
        if y_test[i] == predictions[i]:
            counter += 1
    return round(counter * 100 / y_test.shape[0], 3), counter

def calc_mean_squared_error(y_test, predictions):
    sum = 0
    for i in range(y_test.shape[0]):
        sum += pow((y_test[i] - predictions[i]), 2)
    return round(sum / y_test.shape[0], 3)

def finalize_results(dataset_name, y_test, predictions):
    f = open(f'./data/{dataset_name}-y_test.txt', 'w')
    for y in y_test:
        f.write(f'{y}\n')
    f.close()
    temp_y_test = []
    temp_predictions = []
    for i in range(y_test.shape[0]):
        temp_predictions.append(np.argmax(predictions[i]))
        temp_y_test.append(np.argmax(y_test[i]))
    f = open(f'./data/{dataset_name}-temp_y_test.txt', 'w')
    for y in temp_y_test:
        f.write(f'{y}\n')
    f.close()
    return np.array(temp_y_test), np.array(temp_predictions)

def binarize_results(dataset_name, Y):
    threshold = 12.1
    if dataset_name in DATASETS_TO_PREPROCESS:
        threshold = 7.0
    scores_list = []
    for y in Y:
        if y >= threshold:
            scores_list.append([0, 1])
        else:
            scores_list.append([1, 0])
    return np.array(scores_list)

def measure_and_print_performance(dataset_name, y_test, predictions):
    binary_y_test = binarize_results(dataset_name, y_test)
    binary_predictions = binarize_results(dataset_name, predictions)

    em_ci = emetrics.get_cindex(y_test, predictions)
    em_mse = emetrics.get_k(y_test, predictions)
    em_r2m = emetrics.get_rm2(y_test, predictions)
    em_aupr = emetrics.get_aupr(binary_y_test, binary_predictions)

    mse2 = sk_mse(y_test, predictions)
    # aeg_precision = sk_ap(y_test, predictions)

    save_predictions(dataset_name, y_test, predictions)

    print(f'{dataset_name} dataset:')
    print(f'\tConcordance Index: {em_ci}')
    print(f'\tMean Squared Error: {em_mse}')
    print(f'\tMean Squared Error calculated with SKLearn: {mse2}')
    print(f'\tr2m: {em_r2m}')
    print(f'\tAUPR: {em_aupr}')
    # print(f'\taverage precision: {aeg_precision}')
