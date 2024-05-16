#py -m pip install

import numpy as np
import sklearn as sc
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import projections
import argparse
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn_extra.kernel_approximation import Fastfood

#SVC implementation, takes in given x and y, with kernel if specified, and method of projection
def train_svm(x_train, y_train, x_test, y_test, kern, gamma):
    #add kernel and gamma feature
    svc = SVC(kernel=kern, gamma=gamma, C=1, max_iter=500)
    #fit data
    svc.fit(x_train, y_train)
    #pred
    test = svc.predict(x_test)
    train = svc.predict(x_train)
    #accuracy
    return accuracy_score(train,y_train), accuracy_score(test, y_test)

#inputs
parser = argparse.ArgumentParser()
#verbose mode
parser.add_argument('-dataset', required=True)
args = parser.parse_args()

#load iris dataset
if args.dataset == "iris":
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

#load mnist data set
if args.dataset == "mnist":
    mnist = fetch_openml('mnist_784')
    x = mnist.data
    y = mnist.target 

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2017)

#standard scaling inputs
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#Test Params
scale = 10
dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

#RBF kernel
gamma = 1/(2*scale**2)
train_rbf_acc, test_rbf_acc = train_svm(x_train, y_train, x_test, y_test, "rbf", gamma)
print("Kernel: RBF", "Train_Accuracy:", train_rbf_acc, "Test_Accuracy:", test_rbf_acc)
print("--------------------------------------------------------------------------------")

#RKS implementation
rks_acc_test = []
rks_acc_train = []

for dimension in dimensions:
    #project train
    rks = projections.rks(dimension, scale)
    rks.fit(x_train)
    x_train_rks = rks.transform(x_train)
    #project test
    x_test_rks = rks.transform(x_test)
    #train
    train_acc, test_acc = train_svm(x_train_rks, y_train, x_test_rks, y_test, 'linear', 1)
    rks_acc_test.append(test_acc)
    rks_acc_train.append(train_acc)
    print("Kernel: RKS", "Train_Accuracy:", train_acc, "Test_Accuracy:", test_acc)
    print("--------------------------------------------------------------------------------")

#fastfood
ff_acc_test = []
ff_acc_train = []

for dimension in dimensions:
    #project train
    ff = Fastfood(sigma=scale, n_components=dimension)
    ff.fit(x_train)
    #project test
    x_train_ff = ff.transform(x_train)
    x_test_ff = ff.transform(x_test)
    #train
    train_acc, test_acc = train_svm(x_train_ff, y_train, x_test_ff, y_test, 'linear', 1)
    ff_acc_test.append(test_acc)
    ff_acc_train.append(train_acc)
    print("Kernel: FF", "Train_Accuracy:", train_acc, "Test_Accuracy:", test_acc)
    print("--------------------------------------------------------------------------------")

#plot for test set accuracy
x=[dimensions[0], dimensions[-1]]
y=[test_rbf_acc,test_rbf_acc]
plt.plot(x,y, label='RBF_Kernel')
plt.plot(dimensions, rks_acc_test, label='RKS_Implementation')
plt.plot(dimensions, ff_acc_test, label='FF_Implementation')
plt.xlabel('Dimension (n)')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.savefig('test_acc.png', bbox_inches='tight')


with open("acc_plot.txt", "w") as f:
    for arr in [dimensions, rks_acc_train, rks_acc_test]:
        f.write(' '.join(map(str, arr)) + '\n')                      