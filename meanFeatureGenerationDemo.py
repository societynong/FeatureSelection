from featureUtils import *



def getMeanFeature(sig,exParas):
    fS,f0 = exParas
    return np.mean(sig)

if __name__ == "__main__":
    snr = -55
    featureName = 'Mean'
    nPoint = 50
    dataGeneration(featureName,snr,getMeanFeature,(FS,F0),nPoint)