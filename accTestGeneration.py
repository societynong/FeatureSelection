from featureUtils import *
from sklearn import svm

if __name__ == "__main__":
    snr = -55
    featureName = 'Mean'
    model = svm.SVC(kernel = 'linear')
    testAcc(model,featureName,snr)