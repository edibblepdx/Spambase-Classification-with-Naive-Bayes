# Ethan Dibble
#
# Spambase Classification with Naive Bayes

import numpy as np
import math
from ucimlrepo import fetch_ucirepo 
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def calc_priors(y_train):
    """
    calc priors for class={0,1}
    """
    # unique orders the result
    unique, counts = np.unique(y_train, return_counts=True)

    assert unique[0] == 0
    assert unique[1] == 1

    total = counts[0] + counts[1]

    return counts[0] / total, counts[1] / total

def N(x, mean, std):
    """
    P(x|class)
    """
    y = np.float64((1 / (math.sqrt(2 * np.pi) * std)) * (math.exp(-(((x - mean)**2) / (2 * (std**2))))))
    # return some minimum=10^-8 if y == 0.0
    return y if y != 0.0 else 10**-8

def naive_bayes(x_test, prior_1, prior_0, mean_1, mean_0, std_1, std_0):
    """
    Run Naive Bayes on the test set
    """
    prediction = np.zeros(x_test.shape[0])  # final predictions

    for (x, i) in zip(x_test, range(len(x_test))):
        # prior terms
        a = math.log(prior_1)   # class 1
        b = math.log(prior_0)   # class 0

        for j in range(len(x)):
            # sum over the input features
            a += math.log(N(x[j], mean_1[j], std_1[j])) # class 1
            b += math.log(N(x[j], mean_0[j], std_0[j])) # class 0

        # add the prediction
        prediction[i] = 1 if a > b else 0

    return prediction

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
    mean_1 = np.sum(x_train[0:907], axis=0, dtype="float64") / 907
    mean_0 = np.sum(x_train[907:], axis=0, dtype="float64") / (x_train.shape[0] - 907)
    std_1 = np.std(x_train[0:907], axis=0, dtype="float64")
    std_0 = np.std(x_train[907:], axis=0, dtype="float64")

    # replace all zero std values with some minimal=10^-8
    mask_1 = std_1 == 0.0
    mask_0 = std_0 == 0.0
    std_1[mask_1] = 10**-8
    std_0[mask_0] = 10**-8

    # naive bayes part
    y_pred = naive_bayes(x_test, prior_1, prior_0, mean_1, mean_0, std_1, std_0)

    # Confusion matrices for the perceptron on the test set
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title(f'Naive Bayes Confusion Matrix on Test Set')
    plt.show()

if __name__ == '__main__':
    main()