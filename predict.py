import train
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


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
    return (data_to_std - np.mean(data_to_std, axis=0)) / np.std(data_to_std, axis=0)


def compute_loss(y_true, y_pred):
    m = y_true.shape[1]  # y_true et y_pred doivent être de forme (n_classes, m)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Éviter les log(0)
    loss = -1 / m * np.sum(y_true * np.log(y_pred))
    return loss


def compute_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true[0] == y_pred[0])
    total_predictions = y_true.shape[1]
    return correct_predictions / total_predictions


def compute_recall(y_true, y_pred):
    true_positives = np.sum((y_true[0] == 1) & (y_pred[0] == 1))
    false_negatives = np.sum((y_true[0] == 0) & (y_pred[0] == 1))
    if (true_positives + false_negatives) == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall


def compute_precision(y_true, y_pred):
    true_positives = np.sum((y_true[0] == 1) & (y_pred[0] == 1))
    false_positives = np.sum((y_true[0] == 1) & (y_pred[0] == 0))
    if (true_positives + false_positives) == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    return precision


# DOC :
# https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall


if __name__ == '__main__':
    # test_features = pd.read_csv('datas/test.csv', header=None)
    # type_of_tumor = test_features.iloc[:,1]
    # data = test_features.iloc[:,2:]
    # test_features_std = standardization(data)
    # params = load_model('datas/params.npz')
    # test_Y_bool = [1 if x == 'M' else 0 for x in type_of_tumor]

    # Aucune mesure n'est parfaite à elle seule. Il est donc judicieux d'examiner plusieurs
    # mesures simultanément et de définir le bon équilibre entre précision et rappel.

    # Vrai positif = tumeur correctement classé comme maligne par le modèle.
    # Vrai négatif = tumeur correctement classé comme bénigne par le modèle.

    # Faux positif = tumeur incorrectement classé comme maligne par le modèle. (fausse alerte)
    # Faux négatif = tumeur incorrectement classé comme bénigne par le modèle. (grave)

    # Le rappel est utile lorsque le coût des faux négatifs est élevé

    # precision = true_positives / true_positives + false_positives
    # recall = true_positives / true_positives + false_negatives

    # nb_y_positive = sum(1 for x in type_of_tumor if x == 'M')
    # nb_y_negative = len(test_Y_bool) - nb_y_positive

    # test_Y_bool = np.array(test_Y_bool)
    # test_Y_bool = train.to_one_hot(test_Y_bool, 2).T
    # test_predict = predict(test_features_std.T, params)
    # test_predict = train.to_one_hot(np.array(test_predict), 2)
    # test_predict = np.where(test_predict == 0., 1., 0)
    # print(test_predict.shape)
    #
    # acc = compute_accuracy(test_Y_bool.T, test_predict)
    # recall = compute_recall(test_Y_bool.T, test_predict)
    # precision = compute_precision(test_Y_bool.T, test_predict)
    # accs = accuracy_score(test_Y_bool, test_predict.T)
    # loss = compute_loss(test_Y_bool.T, test_predict)
    # llos = log_loss(test_Y_bool, test_predict.T)
    # print('Real accuracy on total set =', accs)
    # print('Accuracy on total set =', acc)
    # print('---------------------------------')
    # print('Precision on total set =', precision)
    # print('Recall on total set =', recall)
    # print('---------------------------------')
    # print('Real loss on total set =', llos)
    # print('Loss on total set =', loss)
    # print('---------------------------------')

    # test_features = pd.read_csv('datas/train_X_std.csv', header=None)
    # test_bools = pd.read_csv('datas/train_Y_bool.csv', header=None)
    # test_features_std = standardization(test_features)
    # params = load_model('datas/params.npz')
    # test_bools = np.array(test_bools)
    # test_Y_bool = train.to_one_hot(test_bools, 2).T
    #
    # test_predict = predict(test_features_std.T, params)
    # test_predict = train.to_one_hot(np.array(test_predict), 2).T
    # test_predict = np.where(test_predict == 0., 1., 0)
    # print(test_Y_bool.shape)
    # print(test_predict.shape)
    #
    # accs = accuracy_score(test_Y_bool, test_predict)
    # # acc = compute_accuracy(test_Y_bool, test_predict)
    # llos = log_loss(test_Y_bool, test_predict)
    # # loss = compute_loss(test_Y_bool, test_predict)
    # print('Real accuracy on total set =', accs)
    # # print('Accuracy on total set =', acc)
    # print('Real loss on total set =', llos)
    # # print('Loss on total set =', loss)
    # print('---------------------------------')

    test_features = pd.read_csv('datas/validation_X_std.csv', header=None)
    test_bools = pd.read_csv('datas/validation_Y_bool.csv', header=None)
    params = load_model('datas/params.npz')

    test_Y_bool = np.array(test_bools)
    test_Y_bool = train.to_one_hot(test_Y_bool, 2).T

    test_predict = predict(test_features.T, params)
    test_predict = train.to_one_hot(np.array(test_predict), 2)
    test_predict = np.where(test_predict == 0., 1., 0)

    acc = compute_accuracy(test_Y_bool.T, test_predict)
    recall = compute_recall(test_Y_bool.T, test_predict)
    precision = compute_precision(test_Y_bool.T, test_predict)
    accs = accuracy_score(test_Y_bool, test_predict.T)
    loss = compute_loss(test_Y_bool.T, test_predict)
    llos = log_loss(test_Y_bool, test_predict.T)
    print('Real accuracy on total set =', accs)
    print('Accuracy on validation set =', acc)
    print('---------------------------------')
    print('Precision on validation set =', precision)
    print('Recall on validation set =', recall)
    print('---------------------------------')
    print('Real loss on validation set =', llos)
    print('Loss on validation set =', loss)
    print('---------------------------------')
