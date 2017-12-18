import numpy
import math
import csv
from sklearn import linear_model, svm
from sklearn.model_selection import GridSearchCV

import itertools
import numpy
import math
import csv
from sklearn import linear_model
from sklearn import svm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams["font.family"] = "Times New Roman"

def generate_conf_matrix(real, pred, normalize=False, cmap=plt.cm.Blues):
    classes = ['Pop Rock','Electronic','Rap','Jazz','Latin','RnB','International','Country', 'Reggae', 'Blues']
    cm = confusion_matrix(real,pred)
    numpy.set_printoptions(precision=1)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.clim(0,1)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.savefig("foo.png", bbox_inches='tight', dpi=900)
    plt.show()

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

features = import_features('train_data.csv')
y = import_y('train_labels.csv')

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

def export_data(y):
    with open('pred_label.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Sample_id','Sample_label'])
        for i in range(0, len(y)):
            index = i+1
            writer.writerow([str(index),str(int(y[i]))])


# In[6]:


def get_classifier(x, y):
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

    # Train the classifier on data1's feature and target data
    clf.fit(features, y)   
    reg = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(x, y)
    return reg

def classify(reg, x):
    return reg.predict(x)

def calculate_p(reg, x):
    return reg.predict_proba([x])

test_set = import_features('train_data.csv')

classifier = get_classifier(features, y)
y_pred = classify(classifier, test_set)

#export_data(y_pred)
generate_conf_matrix(y, y_pred, True)