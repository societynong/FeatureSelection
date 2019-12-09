from featureUtils import *


def generateTianxianData():
    n = 30
    sigs = []
    ns = []
    ph = 30 * np.pi / 180
    for i in range(n):
        x0,n0 = getSig(60 * 20,F0,FP,FS,-60,i * ph)
        sigs.append(x0 + n0)
        _,n0 = getSig(60 * 20,F0,FP,FS,-60,i * ph)
        ns.append(n0)
        print('{:.2%}'.format(i / n))

    with open('sigs.pkl','wb') as f:
        pkl.dump(sigs,f)

    with open('ns.pkl','wb') as f:
        pkl.dump(ns,f)
def loadTianxianData():
    with open('sigs.pkl','rb') as f:
        sigs = pkl.load(f)

    with open('ns.pkl','rb') as f:
        ns = pkl.load(f)

    return sigs,ns

def getSinglePointF(sig,f0,fs):

    nL = len(sig)
    f = np.arange(nL) / nL * FS
    df = FS / nL

    sigF = np.fft.fft(sig) / len(sig)

    return sigF[np.where(np.abs(f - f0) < df / 2)]

if __name__ == '__main__':
    # generateTianxianData()
    sigs,ns = loadTianxianData()

    sigs = sigs[:16]
    ns = ns[:16]
    sigFs = []
    for sig in sigs:
        sigFs.append(getSinglePointF(sig,F0,FS))
    sigFs = np.array(sigFs)
    nFs = []
    for no in ns:
        nFs.append(getSinglePointF(no,F0,FS))
    nFs = np.array(nFs)
    nA = len(sigs)
    dps = np.arange(0,360,1) / 180 * np.pi
    fix0 = np.exp(-1j * dps)
    fixes = np.zeros((fix0.shape[0],nA),complex)
    for i in range(nA):
        fixes[:,i] = fix0 ** i
    
    resSig = fixes.dot(sigFs)
    resN = fixes.dot(nFs)
    plt.figure()
    # plt.subplot(2,1,1)
    plt.plot(np.abs(resSig))

    # plt.subplot(2,1,2)
    plt.plot(np.abs(resN))
    plt.show()




