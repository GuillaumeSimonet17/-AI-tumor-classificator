from numpy import ndarray
import numpy as np
import pandas as pd
import train


def predict(X, params):
    activations = train.forward_propagation(X, params)
    L = len(params) // 2
    Af = activations['A' + str(L)]
    predictions = np.argmax(Af, axis=0)
    pred = [1 if x == 0 else 0 for x in predictions]
    return pred


def load_model(file_path):
    data = np.load(file_path)
    parameters = {key: data[key] for key in data.files}
    return parameters


def standardization(data_to_std):
    if isinstance(data_to_std, ndarray) and data_to_std.shape[0] >= 2 :
        return (data_to_std - np.mean(data_to_std, axis=0)) / np.std(data_to_std, axis=0)
    else:
        print('Not a ndarray or is a one line array')

def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss


def compute_accuracy(y_true, y_pred):
    y_true_col1 = y_true[:, 0]
    y_pred_col1 = y_pred[:, 0]
    correct_predictions = np.sum(y_true_col1 == y_pred_col1)
    total_predictions = y_true.shape[0]
    return correct_predictions / total_predictions


def compute_recall(y_true, y_pred):
    y_true_col1 = y_true[:, 0]
    y_pred_col1 = y_pred[:, 0]
    true_positives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 1))
    false_negatives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 0))
    # print('true_positives = ', true_positives)
    # print('false_negatives = ', false_negatives)
    if (true_positives + false_negatives) == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall


def compute_precision(y_true, y_pred):
    y_true_col1 = y_true[:, 0]
    y_pred_col1 = y_pred[:, 0]
    true_positives = np.sum((y_true_col1 == 1) & (y_pred_col1 == 1))
    false_positives = np.sum((y_true_col1[0] == 0) & (y_pred_col1[0] == 1))
    if (true_positives + false_positives) == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    return precision


def standardization_with_values(data, std_values):
    means = np.array(std_values[:1])
    std = np.array(std_values[1:2])
    return (data - means) / std


if __name__ == '__main__':
    try:
        std_values = pd.read_csv('datas/std_values.csv', header=None)
        test_features = pd.read_csv('datas/test.csv', header=None)
        params = load_model('datas/params.npz')
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {e}")
        raise

    type_of_tumor = test_features.iloc[:,1]
    data = test_features.iloc[:,2:]
    data = np.array(data)
    test_features_std = standardization_with_values(data, std_values)

    test_Y_bool = [1 if x == 'M' else 0 for x in type_of_tumor]

    test_Y_bool = np.array(test_Y_bool)
    test_Y_bool = train.to_one_hot(test_Y_bool, 2)

    test_predict = predict(test_features_std.T, params)
    test_predict = train.to_one_hot(np.array(test_predict), 2)

    if test_predict.shape == (1,2) and test_predict[0][0] == 0:
        print('====== La tumeur est bénigne')
    elif test_predict.shape == (1,2) and test_predict[0][0] == 1:
        print('====== La tumeur est maligne')

    acc = compute_accuracy(test_Y_bool, test_predict)
    recall = compute_recall(test_Y_bool, test_predict)
    precision = compute_precision(test_Y_bool, test_predict)
    loss = train.compute_loss(test_Y_bool.T, test_predict.T)
    print('Accuracy on total set =', acc)
    print('Precision on total set =', precision)
    print('Recall on total set =', recall)
    print('---------------------------------')
    print('Loss on total set =', loss)
    print('---------------------------------')
    print()

    try:
        test_features = pd.read_csv('datas/validation_X_std.csv', header=None)
        test_bools = pd.read_csv('datas/validation_Y_bool.csv', header=None)
        params = load_model('datas/params.npz')
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {e}")
        raise

    test_Y_bool = np.array(test_bools)
    test_Y_bool = train.to_one_hot(test_Y_bool, 2)

    test_predict = predict(test_features.T, params)
    test_predict = train.to_one_hot(np.array(test_predict), 2)

    acc = compute_accuracy(test_Y_bool, test_predict)
    recall = compute_recall(test_Y_bool, test_predict)
    precision = compute_precision(test_Y_bool, test_predict)
    loss = train.compute_loss(test_Y_bool.T, test_predict.T)

    print('Accuracy on validation set =', acc)
    print('Precision on validation set =', precision)
    print('Recall on validation set =', recall)
    print('---------------------------------')
    print('Loss on validation set =', loss)
    print('---------------------------------')