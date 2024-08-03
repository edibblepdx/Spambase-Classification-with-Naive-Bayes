# Ethan Dibble
#
# Spambase Classification with Naive Bayes

import numpy as np
from ucimlrepo import fetch_ucirepo 

def calc_priors(y_train):
    unique, counts = np.unique(y_train, return_counts=True)

    assert unique[0] == 0
    assert unique[1] == 1

    total = counts[0] + counts[1]

    return counts[0] / total, counts[1] / total

def calc_means(x_train):
    return np.sum(x_train, axis=0, dtype="float64")
  
def main():
    # fetch dataset 
    spambase = fetch_ucirepo(id=94) 
    
    # data (as pandas dataframes) 
    x = spambase.data.features 
    y = spambase.data.targets 
    
    # metadata 
    # print(spambase.metadata) 
    
    # variable information 
    # print(spambase.variables) 

    # split the data into test and train sets
    # the data is given with all class=1 preceding class=0
    # class = 1; elements = [1:1813]
    # class = 0; elements = [1814:4601]
    x_train = np.concatenate((x[0:907], x[1813:3207]))
    y_train = np.concatenate((y[0:907], y[1813:3207]))
    x_test = np.concatenate((x[907:1813], x[3207:]))
    y_test = np.concatenate((y[907:1813], y[3207:]))

    # print the class counts per set and total
    # np.unique reorders the classes 0, 1
    unique, counts = np.unique(y, return_counts=True)
    print('total class counts:', dict(zip(unique, counts)))
    unique, counts = np.unique(y_train, return_counts=True)
    print('train class counts:', dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print('test class counts: ', dict(zip(unique, counts)))

    # calculate the priors for the test set
    prior_0, prior_1 = calc_priors(y_train)

    # calculate the mean and standard deviation for the features of each class
    # in the training set
    # all class=1 precede class=0
    mean_1 = np.sum(x_train[0:907], axis=0, dtype="float64")
    mean_0 = np.sum(x_train[907:], axis=0, dtype="float64")
    std_1 = np.std(x_train[0:907], axis=0, dtype="float64")
    std_0 = np.std(x_train[907:], axis=0, dtype="float64")


if __name__ == '__main__':
    main()