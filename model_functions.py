'''
Implementations of the confusion matrix and the ROC curves were i
nspired and adapted from the examples used on scikit-learn.org.
'''

# _*_ coding: utf-8 _*_
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from helper_functions import read_data, hydrophobicity_encode

from collections import defaultdict

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


import itertools
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_substring_count(s, sub_s):
    '''
    Finds the overlapping count of a substring, sub_s, within a string, s
    '''
    return sum(1 for m in re.finditer('(?=%s)' % sub_s, s))


# _*_ coding: utf-8 _*_
amino_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                 'Y', 'U', 'X']


def sequence_features(sequence):
    '''
    Given a sequence, extracts features and stores them in a dictionary
    '''
    features = defaultdict(float)

    features['length'] = len(sequence)
    
    pa = ProteinAnalysis(sequence)

    n = 50
    pa_f50 = ProteinAnalysis(sequence[:n])
    pa_l50 = ProteinAnalysis(sequence[-n:])

    amino_count_dict = pa.get_amino_acids_percent()
    amino_count_dict_f50 = pa_f50.get_amino_acids_percent()
    amino_count_dict_l50 = pa_l50.get_amino_acids_percent()
    
    for amino1 in amino_letters:
        if amino1 in amino_count_dict.keys():
            features["Percentage: " + amino1] = amino_count_dict[amino1]
        else:
            features["Percentage: " + amino1] = 0

        if amino1 in amino_count_dict_f50.keys():
            features["First(50)%: " + amino1] = amino_count_dict_f50[amino1]
        else:
            features["First(50)%: " + amino1] = 0

        if amino1 in amino_count_dict_l50.keys():
            features["Last(50)%: " + amino1] = amino_count_dict_l50[amino1]
        else:
            features["Last(50)%: " + amino1] = 0

        for amino2 in amino_letters:
            features["Percentage: " + amino1 + amino2] = float(
                get_substring_count(sequence, amino1 + amino2)) / (len(sequence) - 1)
            features["First(50)%: " + amino1 + amino2] = float(get_substring_count(
                sequence[:50], amino1 + amino2)) / (len(sequence[:50]) - 1)
            features["Last(50)%: " + amino1 + amino2] = float(get_substring_count(
                sequence[-50:], amino1 + amino2)) / (len(sequence[-50:]) - 1)

    sequence = sequence.replace('X', '')
    sequence = sequence.replace('U', '')
    sequence = sequence.replace('B', 'D') if np.random.random(
    ) < 0.5 else sequence.replace('B', 'N')

    pa = ProteinAnalysis(sequence)

    pa_f50 = ProteinAnalysis(sequence[:n])
    pa_l50 = ProteinAnalysis(sequence[-n:])

    features['hydrophobicity'] = hydrophobicity_encode(sequence)
    features['hydrophobicity (first 50)'] = hydrophobicity_encode(
        sequence[:50])
    features['hydrophobicity (last 50)'] = hydrophobicity_encode(
        sequence[-50:])

    features['aromicity'] = pa.aromaticity()
    features['instability_index'] = pa.instability_index()

    features['isoelectric_point'] = pa.isoelectric_point()
    features['isoelectric_point (first 50)'] = pa_f50.isoelectric_point()
    features['isoelectric_point (last 50)'] = pa_l50.isoelectric_point()

    helix, turn, sheet = pa.secondary_structure_fraction()
    helix_f, turn_f, sheet_f = pa_f50.secondary_structure_fraction()
    helix_l, turn_l, sheet_l = pa_l50.secondary_structure_fraction()

    features['Helix'] = helix
    features['Turn'] = turn
    features['Sheet'] = sheet

    features['Helix (first 50)'] = helix_f
    features['Turn (first 50)'] = turn_f
    features['Sheet (first 50)'] = sheet_f

    features['Helix (last 50)'] = helix_l
    features['Turn (last 50)'] = turn_l
    features['Sheet (last 50)'] = sheet_l

    features['molecular_weight'] = pa.molecular_weight()
    features['molecular_weight (first 50)'] = pa_f50.molecular_weight()
    features['molecular_weight (last 50)'] = pa_l50.molecular_weight()

    features['gravy'] = pa.gravy()
    
    return features

def train(X_data, y_data, x_test, y_test):
    '''
    Arguments:
        X_data: protein training sequences
        y_data: protein training labels
        x_test: protein test sequences
        y_test: protein test labels
    Returns:
        model: a trained model using XGBoost with 1000 estimators
               with a max depth of 15
        vectorizer: the DictVectorizer in order to retrieve feature
                    names
    '''
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()

    train_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in X_data])
    train_sequence_y = label_encoder.fit_transform([y for y in y_data])

    test_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in x_test])
    test_sequence_y = label_encoder.fit_transform([y for y in y_test])

    model = XGBClassifier(n_estimators=10, max_depth=2, nthread=-1)

    model.fit(train_sequence_x, train_sequence_y)

    y_pred = model.predict(test_sequence_x)

    accuracy = model.score(test_sequence_x, test_sequence_y)
    precision = precision_score(test_sequence_y, y_pred, average='weighted')
    recall = recall_score(test_sequence_y, y_pred, average='weighted')
    f1_s = f1_score(test_sequence_y, y_pred, average='weighted')

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1 Score: " + str(f1_s))

    return model, vectorizer

def class_feature_importances(X_train, y_train):
    '''
    Arguments:
        X_train: Protein sequences
        y_train: Protein labels

    Plots a bar graph of the feature importances
    for each class.
    '''
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()

    train_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in X_train])
    train_sequence_y = label_binarize(label_encoder.fit_transform(
        [y for y in y_train]), classes=[0, 1, 2, 3])

    model = OneVsRestClassifier(XGBClassifier(
        n_estimators=1000, max_depth=15, nthread=-1))
    model.fit(train_sequence_x, train_sequence_y)

    for i in range(4):
        feature_importances = model.estimators_[i].feature_importances_
        features_names = vectorizer.get_feature_names()
        features = sorted(zip(feature_names, feature_importances),
                          key=lambda x: x[1], reverse=True)[:30]

        class_title = ['Cytosolic', 'Mitochondrial', 'Nuclear', 'Secreted']

        plt.title(class_title[i])
        plt.bar(range(len(features)), [feature[1]
                                       for feature in features], align='center')
        plt.xticks(range(len(features)), [feature[0]
                                          for feature in features], rotation='vertical')
        plt.xlabel("Features")
        plt.ylabel("Feature Importances")
        plt.title("Top 30 most important features (" +
                  str(class_title[i]) + ")")

        plt.show()

def roc_curve_evaluate(X_train, y_train, X_test, y_test, kfold=True):
    '''
    Calculates and plots the ROC curves for either 10 fold cross
    validation or on the test set depending on the boolean kfold.
    '''
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()

    train_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in X_train])
    train_sequence_y = label_encoder.fit_transform([y for y in y_train])

    if kfold == True:
        skf = StratifiedKFold(n_splits=10)

        roc_list = []
        fpr_list = []
        tpr_list = []

        for train_indices, test_indices in skf.split(train_sequence_x, train_sequence_y):
            x_train = train_sequence_x[train_indices]
            y_train = label_binarize(
                train_sequence_y[train_indices], classes=[0, 1, 2, 3])

            x_test = train_sequence_x[test_indices]
            y_test = label_binarize(
                train_sequence_y[test_indices], classes=[0, 1, 2, 3])

            n_classes = y_train.shape[1]

            # Learn to predict each class against the other
            classifier = OneVsRestClassifier(XGBClassifier(
                n_estimators=1000, max_depth=15, nthread=-1))
            y_score = classifier.fit(x_train, y_train).predict_proba(x_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                base_fpr = np.arange(0, 1, 0.005)
                fprr, tprr, _ = roc_curve(y_test[:, i], y_score[:, i])

                tpr[i] = np.interp(base_fpr, fprr, tprr)

                roc_auc[i] = auc(fprr, tprr)

            # Compute micro-average ROC curve and ROC area
            fprr, tprr, _ = roc_curve(y_test.ravel(), y_score.ravel())
            tpr['micro'] = np.interp(base_fpr, fprr, tprr)

            roc_auc["micro"] = auc(fprr, tprr)

            fpr_list.append(base_fpr)
            tpr_list.append(tpr)
            roc_list.append(roc_auc)

        roc_values = [roc.values() for roc in roc_list]
        roc_means = np.mean(roc_values, axis=0)
        roc_stds = np.std(roc_values, axis=0)

        tpr_values = [tpr.values() for tpr in tpr_list]
        tpr_means = np.mean(tpr_values, axis=0)
        tpr_std = np.std(tpr_values, axis=0)

        tprs_upper = np.minimum(tpr_means + 2 * tpr_std, 1)
        tprs_lower = tpr_means - 2 * tpr_std

        plt.figure()
        lw = 2

        plt.plot(base_fpr, tpr_means[0],
                 lw=lw, label='Cytosolic ROC curve (area = %0.3f +/- %0.3f)' % (roc_means[0], 2 * roc_stds[0]))

        plt.fill_between(base_fpr, tprs_lower[0], tprs_upper[0], alpha=0.3)

        plt.plot(base_fpr, tpr_means[1],
                 lw=lw, label='Mitochondrial ROC curve (area = %0.3f +/- %0.3f)' % (roc_means[1], 2 * roc_stds[1]))

        plt.fill_between(base_fpr, tprs_lower[1], tprs_upper[1], alpha=0.3)

        plt.plot(base_fpr, tpr_means[2],
                 lw=lw, label='Nuclear ROC curve (area = %0.3f +/- %0.3f)' % (roc_means[2], 2 * roc_stds[2]))

        plt.fill_between(base_fpr, tprs_lower[2], tprs_upper[2], alpha=0.3)

        plt.plot(base_fpr, tpr_means[3],
                 lw=lw, label='Secreted ROC curve (area = %0.3f +/- %0.3f)' % (roc_means[3], 2 * roc_stds[3]))

        plt.fill_between(base_fpr, tprs_lower[3], tprs_upper[3], alpha=0.3)

        plt.plot(base_fpr, tpr_means[4], color='darkorange',
                 lw=lw, label='ROC curve (micro) (area = %0.3f +/- %0.3f)' % (roc_means[4], 2 * roc_stds[4]))

        plt.fill_between(base_fpr, tprs_lower[4], tprs_upper[
                         4], color='darkorange', alpha=0.3)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic plot (2 std)')
        plt.legend(loc="lower right")
        plt.show()

    else:
        train_sequence_y = label_binarize(
            train_sequence_y, classes=[0, 1, 2, 3])

        vectorizer = DictVectorizer()
        label_encoder = LabelEncoder()

        X_test = vectorizer.fit_transform(
            [sequence_features(x) for x in X_test])

        y_test = label_binarize(label_encoder.fit_transform(
            [y for y in y_test]), classes=[0, 1, 2, 3])

        n_classes = train_sequence_y.shape[1]

        print(n_classes)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(
            XGBClassifier(n_estimators=1000, max_depth=15))
        y_score = classifier.fit(
            train_sequence_x, train_sequence_y).predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0],
                 lw=lw, label='Cytosolic ROC curve (area = %0.2f)' % roc_auc[0])

        plt.plot(fpr[1], tpr[1],
                 lw=lw, label='Mitochondrial ROC curve (area = %0.2f)' % roc_auc[1])

        plt.plot(fpr[2], tpr[2],
                 lw=lw, label='Nuclear ROC curve (area = %0.2f)' % roc_auc[2])

        plt.plot(fpr[3], tpr[3],
                 lw=lw, label='Secreted ROC curve (area = %0.2f)' % roc_auc[3])

        plt.plot(fpr['micro'], tpr['micro'], color='darkorange',
                 lw=lw, label='ROC curve (micro) (area = %0.2f)' % roc_auc['micro'])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic plot')
        plt.legend(loc="lower right")
        plt.show('test_roc2')

def confusion_matrix_evaluate(model, X_test, y_test):
    '''
    Given a model and some test data, this method 
    displays a confusion matrix between the classes.
    '''
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()

    test_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in X_test])
    test_sequence_y = label_encoder.fit_transform([y for y in y_test])

    y_pred = model.predict(test_sequence_x)

    cm = confusion_matrix(test_sequence_y, y_pred)

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

    normalize = True

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    classes = ["Cytosolic", "Mitochondrial", "Nuclear", "Secreted"]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.show()


def kfold_evaluate(X_train, y_train):
    '''
    Arguments:
        X_train: protein training sequences
        y_train: protein training labels

    Performs a K fold (10) cross validation.
    '''
    print("Detailed classification report:")

    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()

    train_sequence_x = vectorizer.fit_transform(
        [sequence_features(x) for x in X_train])
    train_sequence_y = label_encoder.fit_transform([y for y in y_train])

    skf = StratifiedKFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_indices, test_indices in skf.split(train_sequence_x, train_sequence_y):
        x_train = train_sequence_x[train_indices]
        y_train = train_sequence_y[train_indices]

        x_test = train_sequence_x[test_indices]
        y_test = train_sequence_y[test_indices]

        model = XGBClassifier(n_estimators=1000, max_depth=15, nthread=-1)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = model.score(x_test, y_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1_s = f1_score(y_test, y_pred, average='weighted')

        print(accuracy, precision, recall, f1_s)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_s)

    print('Accuracy: %s +/- %s' %
          (np.mean(accuracies), 2 * np.std(accuracies)))
    print('Precision: %s +/- %s' %
          (np.mean(precisions), 2 * np.std(precisions)))
    print('Recall: %s +/- %s' % (np.mean(recalls), 2 * np.std(recalls)))
    print('F1 Score: %s +/- %s' % (np.mean(f1_scores), 2 * np.std(f1_scores)))

def feature_importances(model, vectorizer):
    '''
    Arguments:
        model: trained model
        vectorizer: DictVectorizer object used to store features.
    '''
    feature_names = sorted(vectorizer.feature_names_)
    feature_importances = model.feature_importances_

    features = sorted(zip(feature_names, feature_importances),
                      key=lambda x: x[1], reverse=True)[:30]

    plt.bar(range(len(features)), [feature[1]
                                   for feature in features], align='center')
    plt.xticks(range(len(features)), [feature[0]
                                      for feature in features], rotation='vertical')
    plt.xlabel("Features")
    plt.ylabel("Feature Importances")
    plt.title("Top 30 most important features (All)")

    plt.show()
