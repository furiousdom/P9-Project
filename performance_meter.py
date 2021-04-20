import numpy as np
import emetrics
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

def finalize_results(y_test, predictions):
    temp_y_test = []
    temp_predictions = []
    for i in range(y_test.shape[0]):
        temp_predictions.append(np.argmax(predictions[i]))
        temp_y_test.append(np.argmax(y_test[i]))
    return np.array(temp_y_test), np.array(temp_predictions)

def measure_and_print_performance(dataset_name, y_test, predictions):
    y_test, predictions = finalize_results(y_test, predictions)
    acc, counter = calc_accuracy(y_test, predictions)
    ci = round(emetrics.get_cindex(y_test, predictions), 3)
    mse = round(emetrics.get_k(y_test, predictions), 3)
    r2m = round(emetrics.get_rm2(y_test, predictions), 3)
    aupr = round(emetrics.get_aupr(y_test, predictions), 3)

    print(f'{dataset_name} dataset:')
    print(f'\tAccuracy: {acc}%, predicted {counter} out of {y_test.shape[0]}')
    print(f'\tConcordance Index: {ci}')
    print(f'\tMean Squared Error: {mse}')
    print(f'\tr2m: {r2m}')
    print(f'\tAUPR: {aupr}')
