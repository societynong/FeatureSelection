from featureUtils import *
from sklearn import svm

if __name__ == "__main__":
    snr = -50
    featureName = 'SinglePoint'
    model = MLPClassifier()
    # model = svm.SVC(kernel='linear')
    testAcc(model,featureName,snr)