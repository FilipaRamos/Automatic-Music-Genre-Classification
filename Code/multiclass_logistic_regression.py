import numpy
import math
import csv
from sklearn import linear_model, preprocessing

nr_genres = 10

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

def export_data_accuracy(y):
    with open('pred_label_accuracy.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Sample_id','Sample_label'])
        for i in range(0, len(y)):
            index = i+1
            writer.writerow([str(index),str(y[i])])

def export_data_log_loss(y):
    with open('pred_label_log_loss.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Sample_id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9','Class_10'])
        for i in range(0, len(y)):
            index = i+1
            temp = [str(index)]
            for k in range(0, len(y[i])):
                temp.append(str(y[i][k]))
            writer.writerow(temp)

def get_classifier(x, y):
    reg = linear_model.LogisticRegressionCV(fit_intercept=False, Cs=numpy.logspace(2,8,6), multi_class='multinomial')
    reg.fit(x, y)
    return reg

def classify(reg, x):
    return reg.predict(x)

def calculate_p(reg, x):
    return reg.predict_proba(x)

def filter_data(y, genre):
    y_temp = []
    for i in range(0, len(y)):
        if y[i] == genre:
            y_temp.append(1)
        else:
            y_temp.append(0)
    return y_temp

def scale_features(Xtrain, Xtest):
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest

    # single logistic regression classifier, with feature scaling
def accuracy_test(scaling=False):
    trainSet = import_features('train_data.csv')
    trainLabels = import_y('train_labels.csv')
    testSet = import_features('test_data.csv')
    
    if scaling:
        trainSet, testSet = scale_features(trainSet, testSet)

    classifier = get_classifier(trainSet, trainLabels)
    yPred = classify(classifier, testSet)

    export_data_accuracy(yPred)


def log_loss_test(scaling=False):
    trainSet = import_features('train_data.csv')
    trainLabels = import_y('train_labels.csv')
    testSet = import_features('test_data.csv')
    
    if scaling:
        trainSet, testSet = scale_features(trainSet, testSet)

    classifier = get_classifier(trainSet, trainLabels)

    yPred = calculate_p(classifier, testSet)
    
    export_data_log_loss(yPred)


def main():
    accuracy_test(True)
    print("accuracy_test")
    accuracy_test(False)
    print("accuracy_test")
    log_loss_test(True)
    print("log_loss_test")
    log_loss_test(False)
    print("log_loss_test")

if __name__ == "__main__":
    main()