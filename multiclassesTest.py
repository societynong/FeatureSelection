from featureUtils import *

def getMulticlassFeature(sig,exParas):
    fs,f0s = exParas
    sigF = np.fft.fft(sig)[:len(sig) // 2] / len(sig)
    df = fs / len(sig)
    f = np.arange(len(sig) // 2) / len(sig) * fs
    feature = []
    for f0 in f0s:
        cmpx = sigF[np.where(np.abs(f - f0) < df / 2)]
        feature.extend(np.real(cmpx).tolist())
        feature.extend(np.imag(cmpx).tolist())
    return np.array(feature)



def getData():
    snr = -60
    f0s = [100]
    featureName = 'Multiclass2'
    nPoint = 20
    dataGenerationV2(featureName,snr,getMulticlassFeature,(FS,f0s),f0s,nPoint)


if __name__ == "__main__":
    getData()