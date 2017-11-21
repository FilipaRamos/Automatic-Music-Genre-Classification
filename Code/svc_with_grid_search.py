
# coding: utf-8
import numpy
import math
import csv
from sklearn import linear_model, svm
from sklearn.model_selection import GridSearchCV


def import_features(filepath):
    data = []
    with open(filepath, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        for row in file:
            line = []
            for i in range(0, 264):
                line.append(float(row[i]))
            data.append(line)
        return data


def import_y(filepath):
    return numpy.loadtxt(filepath)


def export_data(y):
    with open('pred_label.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Sample_id','Sample_label'])
        for i in range(0, len(y)):
            index = i+1
            writer.writerow([str(index),str(int(y[i]))])


def get_classifier(x, y):
    parameter_candidates = [
      {'C': numpy.logspace(2,8,6), 'gamma': numpy.logspace(-7,-1,6), 'kernel': ['rbf']},
    ]

    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

    # Train the classifier on data1's feature and target data
    clf.fit(x, y)
    reg = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(x, y)
    return reg

def classify(reg, x):
    return reg.predict(x)


def calculate_p(reg, x):
    return reg.predict_proba(x)


def main():
    features = import_features('train_data.csv')
    y = import_y('train_labels.csv')

    test_set = import_features('test_data.csv')
    classifier = get_classifier(features, y)
    y_pred = classify(classifier, test_set)

    export_data(y_pred)


if __name__ == "__main__":
	main()
