from featureUtils import *

if __name__ == "__main__":
    snr = -45
    featureName = 'SinglePoint'
    nPoint = 50
    dataGeneration(featureName,snr,getSiglePointFeature,(FS,F0),nPoint)