# classifiers to import:
#  - SVM-Linear
#  - SVM-RBF
#  - SVM-Quadratic
#  - Naive Bayes

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def classification_benchmarks(train_data, eval_data):
    # train_data is a list of tuples of (df_filtered, age_label)
    # eval_data is a list of tuples of (df_filtered, age_label)

    # create a list of references to the classifiers we want to use
    classifiers = [svm.SVC(kernel='linear'), svm.SVC(
        kernel='rbf'), svm.SVC(kernel='poly'), GaussianNB()]

    # create a list of names for the classifiers
    classifier_names = ["SVM-Linear", "SVM-RBF",
                        "SVM-Quadratic", "Naive Bayes"]

    # create a list to hold the accuracy scores for each classifier
    accuracy_scores = []

    # create a list to hold the confusion matrices for each classifier
    confusion_matrices = []

    # create a list to hold the classification reports for each classifier
    classification_reports = []

    # create a list to hold the confusion matrix plots for each classifier
    confusion_matrix_plots = []

    # loop through each classifier, perform the training and evaluation, and store the results
    for classifier in classifiers:
        # create a list to hold the training data
        train_data_list = []

        # create a list to hold the training labels
        train_labels_list = []

        # create a list to hold the evaluation data
        eval_data_list = []

        # create a list to hold the evaluation labels
        eval_labels_list = []

        # loop through each tuple in the train data
        for df in train_data:
            # add the dataframe to the list of training data
            train_data_list.append(df[0])

            # add the age label to the list of training labels
            train_labels_list.append(df[1])

        # loop through each tuple in the eval data
        for df in eval_data:
            # add the dataframe to the list of evaluation data
            eval_data_list.append(df[0])

            # add the age label to the list of evaluation labels
            eval_labels_list.append(df[1])

        # concatenate the training data into one dataframe
        train_data_df = pd.concat(train_data_list)

        # concatenate the training labels into one dataframe
        train_labels_df = pd.concat(train_labels_list)

        # concatenate the evaluation data into one dataframe
        eval_data_df = pd.concat(eval_data_list)

        # concatenate the evaluation labels into one dataframe
        eval_labels_df = pd.concat(eval_labels_list)

        # train the classifier
        classifier.fit(train_data_df, train_labels_df)

        # evaluate the classifier
        predictions = classifier.predict(eval_data_df)

        # calculate the accuracy score
        accuracy_scores.append(accuracy_score(eval_labels_df, predictions))

        # calculate the confusion matrix
        confusion_matrices.append(
            confusion_matrix(eval_labels_df, predictions))

        # calculate the classification report
        classification_reports.append(
            classification_report(eval_labels_df, predictions))

        # calculate the confusion matrix plot
        confusion_matrix_plots.append(
            plot_confusion_matrix(classifier, eval_data_df, eval_labels_df))

        return accuracy_scores, confusion_matrices, classification_reports, confusion_matrix_plots
