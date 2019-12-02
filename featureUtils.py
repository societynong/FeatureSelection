import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import sys
from sklearn.model_selection import train_test_split
DUR = 60*20
FS = 10000
N = int(DUR * FS)
t = np.arange(N) / FS
F0 = 100
FP = 300


def srU(a,b,x):
    return -a*x + b * x**3

def srDuf(a,b,k,h,sig,f):
    y = np.zeros(len(sig))
    x = np.zeros(len(sig))
    for i in range(len(y) - 1):
        K1 = h * y[i]
        L1 = h * (-k * y[i] - f(a,b,x[i]) + sig[i])
        K2 = h * (y[i] + L1 / 2)
        L2 = h * (-k * (y[i] + L1 / 2) - f(a,b,x[i] + K1 / 2) + sig[i])
        K3 = h * (y[i] + L2 / 2)
        L3 = h * (-k * (y[i] + L2 / 2) - f(a,b,x[i] + K2 / 2) + sig[i + 1])
        K4 = h * (y[i] + L3)
        L4 = h * (-k * (y[i] + L3) - f(a,b,x[i] + K3) + sig[i + 1])
        x[i + 1] = x[i] + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        y[i + 1] = y[i] + 1 / 6 * (L1 + 2 * L2 + 2 * L3 + L4)
    return x

def srFun(a,b,h,sig):

    u = np.zeros(len(sig))

    for i in range(len(u) - 1):
        k1 = h * (a * u[i] - b * u[i] ** 3 + sig[i])
        k2 = h * (a * (u[i] + k1 / 2) - b * (u[i] + k1 / 2) ** 3 + sig[i])
        k3 = h * (a * (u[i] + k2 / 2) - b * (u[i] + k2 / 2) ** 3 + sig[i + 1])
        k4 = h * (a * (u[i] + k3) - b * (u[i] + k3) ** 3 + sig[i + 1])
        u[i + 1] = u[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # u = u - np.mean(u)
    return u

def srFunMine(a,b,h,sig):
    u = np.zeros(len(sig))
    for i in range(1,len(u)):
        u[i] =u[i - 1] + h * (a * u[i - 1] - b * u[i - 1] ** 3 + sig[i - 1])
    return u
def getSig(dur,f0,fP,fS,snr):
    t = np.arange(dur * fS) / fS
    x0 = np.sin(2*np.pi*f0 * t)
    ns = np.random.randn(len(t))
    # ns = funs.butter_filter(ns,10,FP,FS,'lowpass')
    nsF = np.fft.fft(ns)
    df = fS / len(ns)
    f = np.arange(len(ns)) / len(ns) * fS
    fPN = np.where(np.abs(f - fP) < df / 2)[0][0]
    nsF[fPN + 1 : len(nsF) - fPN] = np.zeros(len(nsF[fPN + 1 : len(nsF) - fPN]))
    ns = np.real(np.fft.ifft(nsF))
    ns = ns - np.mean(ns)

    sigPower = 1 / len(t) * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(snr / 10))
    ns = np.sqrt(nsiPower) / np.std(ns) * ns

    return x0 , ns

def getNoiseFix(t,F0,FP,FS,SNR):
    x0 = np.sin(2*np.pi*F0*t)
    ns = np.random.randn(len(t))
    ns = funs.butter_filter(ns,3,FP,FS,'highpass')
    sigPower = 1 / N * np.sum(x0 * x0)

    nsiPower = sigPower / (10**(SNR / 10))

    ns = np.sqrt(nsiPower) / np.std(ns) * ns
    return x0 , ns

def getAcmSig(sig,W,S):
    sigAcm = 0
    for i in range(0,len(sig) - W,S):
        sigAcm = sigAcm + sig[i:i+ W]
    return sigAcm / len(range(0,len(sig) - W,S))
def showInF(sig,fMax,fS):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * fS
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))

def showInFDetail(sig,fMax,fS,fGoal):
    NFFT = len(sig)
    F_SHOW = int(fMax // (fS / NFFT))
    F_ABS = np.abs(np.fft.fft(sig)[:NFFT // 2]) / len(sig)
    f = np.arange(F_SHOW) / NFFT * FS
    f0 = np.where(np.abs(f - fGoal) < fS / NFFT / 2 )
    plt.plot(f[:F_SHOW],F_ABS[:F_SHOW])
    plt.xlabel("Hz")
    plt.title("Signal weight:{} Highest frequency location:{}Hz".format(F_ABS[f0] * (len(F_ABS) - 1) / ((np.sum(F_ABS) - F_ABS[f0])),f[np.argmax(F_ABS[:F_SHOW])]))
    # plt.title("Max F:{}".format(f[np.argmax(F_ABS[:F_SHOW])]))

def showInT(sig,FS):
    t = np.arange(len(sig)) / FS
    plt.plot(t,sig)
    plt.xlabel("s")


def goalFunc(ab,FS,x0,n0):
    a,b = ab
    sr = srFun(a,b,1 / FS,x0 + n0)
    srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    f0N = int(F0 / (FS / len(sr)))

    # sigF = np.abs(np.fft.fft(x0 + n0))[:int(len(x0 + n0) / 2)]

    return srF[f0N] / np.mean(np.hstack((srF[:f0N],srF[f0N:])))
    # return 1 / (srF[f0N] / np.mean(np.hstack((srF[:f0N],srF[f0N:]))) / (sigF[f0N] / np.mean(np.hstack((sigF[:f0N],sigF[f0N:])))))
    # return -np.mean(sr*x0)**2

def goalFuncPSO(ab):
    a,b = ab
    x0 = A * np.sin(2 * np.pi * F0 * t)
    n0 = np.sqrt(2 * D) * np.random.standard_normal(len(x0))
    sr = srFun(a,b,1 / FS,x0 + n0)
    # srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    # f0N = int(F0 / (FS / len(sr)))
    # return -srF[f0N] / sum(srF)
    return -np.mean(sr*x0)**2

def goalFuncPSOA(a):
    x0 = 0.01 * np.sin(2 * np.pi * F0 * t)
    n0 = np.sqrt(2 * D) * np.random.standard_normal(len(x0))
    sr = srFun(a,1,1 / FS,x0 + n0)
    # srF = np.abs(np.fft.fft(sr))[:int(len(sr) / 2)]
    # f0N = int(F0 / (FS / len(sr)))
    # return -srF[f0N] / sum(srF)
    return -np.mean(sr*x0)**2

def selIMF(imfs,sig):
    maxV = 0
    sig = sig - np.mean(sig)
    for i in range(imfs.shape[0]):
        imfI = imfs[i]
        imfI = imfI - np.mean(imfI)
        goal = np.sum(imfI * sig) / np.sqrt(np.sum(imfI ** 2) * np.sum(sig ** 2))
        if goal > maxV:
            maxV = goal
    return imfI[i]


def snr(x,n):
    return 10 * np.log10(np.sum(x**2) / np.sum(n ** 2))
def getSnr(x,n):
    return 10 * np.log10(np.sum(x**2) / np.sum(n ** 2))

def lineMap(mi,mx,sig): 
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * (mx - mi) + mi



def getMaxF(sig,FS):
    NFFT = len(sig)
    f = np.arange(NFFT) / NFFT * FS
    sigF = np.abs(np.fft.fft(sig))[:int(NFFT // 2)]
    return f[np.argmax(sigF)]





def jiangpin(sig,f0,f1):
    n = int(f0 / f1)
    m = int(len(sig) / n)
    ret = sig[:m*n]
    ret = np.reshape(sig,(n,m))
    ret = np.transpose(ret,(1,0))
    ret = np.reshape(ret,m*n)
    return ret

def getAcmHalFeature(sig,exParas):
    fS,f0 = exParas
    sig = sig[:int(len(sig) / f0) * f0]
    W = int(fS / f0 // 2)
    feature = 0
    flag = 1
    for i in range(0 , len(sig) - W, W):
        feature += flag * sig[i:i+W] 
        flag *= -1
    return (feature / len(range(0 , len(sig) - W, W)))


def getSiglePointFeature(sig,exParas):
    fS,f0 = exParas
    sigF = np.fft.fft(sig)
    df = fS / len(sigF)
    f = np.arange(len(sig)) / len(sig) * FS
    f2N = np.where(np.abs(f - f0) < df / 2)[0][0]
    sigF0 = np.zeros(len(sigF),dtype=complex)
    sigF0[f2N] = sigF[f2N]
    sigF0[-f2N] = sigF[-f2N]
    
    featureSiglePoint = np.real(np.fft.ifft(sigF0))
    featureSiglePoint = getAcmSig(featureSiglePoint,200,200)
    return featureSiglePoint

def dataGeneration(featureName,snr,featureFun,exParas,nPoints = 200):
    if not os.path.exists("features\\{}".format(featureName)):
        os.makedirs("features\\{}".format(featureName))
    filetoSave = "features\\{}\\{}.pkl".format(featureName,snr)
    fN0Rec = []
    fSigRec = []
    for i in range(nPoints):
        x0, n0 = getSig(DUR, F0, FP, FS, snr)
        # fN0 = getSiglePointFeature(n0,exParas)
        fN0 = featureFun(n0,exParas).tolist()
        x0, n0 = getSig(DUR, F0, FP, FS, snr)
        # fSig = getSiglePointFeature(x0 + n0,FS,F0)
        fSig = featureFun(x0+n0,exParas).tolist()
        fN0Rec.append(fN0)
        fSigRec.append(fSig)
        print("{:.2f}% of {}db".format(i / nPoints * 100,snr))
    
    XList = fN0Rec+fSigRec
    yList = ([0] * len(fN0Rec))+([1] * len(fSigRec))

    if os.path.exists(filetoSave):
        with open(filetoSave,'rb') as f:
            XOld,yOld = pkl.load(f)
        XList += XOld.tolist()
        yList += yOld.tolist()
        

    X = np.array(XList)
    y = np.array(yList)
    with open(filetoSave,'wb') as f:
        pkl.dump((X,y),f)


from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.model_selection import cross_val_score
# from deepnetwork import FeatureNet
def testSinglePoint(start,stop,step,featureName):
    # nPoints = 40
    # plt.figure()
    # plt.title("SNR:{}".format(SNR))
    # fN0Rec = []
    # fSigRec = []
    # for i in range(nPoints):
    #     x0, n0 = getSig(t, F0, FP, FS, SNR)
    #     fN0 = _testPoint(n0,F0)
    #     fX0 = _testPoint(x0,F0)
    #     fSig = _testPoint(x0 + n0,F0)
    #     # print("x0:{},n0:{},x0 + n0:{}".format(fX0,fN0,fSig))
    #     fN0Rec.append(fN0)
    #     fSigRec.append(fSig)
    #     plt.scatter(fN0,fN0,c='red')
    #     plt.scatter(fSig,fSig,c='green')
    #     print("{}%".format(i / nPoints * 100))

    # # plt.savefig("{}db.png".format(SNR))
    # plt.show()
    # X = np.reshape(np.array(fN0Rec+fSigRec),[-1,1])
    # y = np.array(([0] * len(fN0Rec))+([1] * len(fSigRec)))
    # model = svm.SVC(kernel='linear')

    for snr in range(start,stop,step):
        with open("features\\test\\{}{}.pkl".format(snr,featureName),'rb') as f:
            X,y = pkl.load(f)
        rIdx = np.array(list(range(len(X))))
        np.random.shuffle(rIdx)
        X = X[rIdx]
        y = y[rIdx]
        # fnw = FeatureNet().cuda()
        # StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=3)
        # fnw.fit(X_train.reshape([X_train.shape[0],1,X_train.shape[1],X_train.shape[2]]),y_train)
        # # X = fnw(np.concatenate(X.reshape([X.shape[0],1,X.shape[1],X.shape[2]])))
        # y_pred = fnw(X_test.reshape([X_test.shape[0],1,X_test.shape[1],X_test.shape[2]])).cpu().detach().numpy()
        # print(y_pred)
        # print(y_test)
        # print("Accuracy:{}".format(np.mean(y_pred == y_test)))
      
        
        
        model = svm.SVC(kernel='linear',gamma='auto')#tree.DecisionTreeClassifier()
        # model = MLPClassifier()
        model.fit(X_train,y_train)
        acc = cross_val_score(model,X_test,y_test,cv=5,scoring='accuracy')#model.score(X_test,y_test)
        print("{} db, Acc:{},Mean:{}".format(snr,acc,np.mean(acc)))

from scipy.signal import stft
import cv2
# from scipy import misc

def getSTFTFeature(sig,fS,win):
    f, tt, Zxx = stft(sig, fS, nperseg=win)
    Zxx = abs(Zxx)
    df = fS / win
    f = f[:int(FP / df) ]
    Zxx = Zxx[:int(FP / df) ][:]
    # imgZxx = np.uint16(lineMap(0,2 ** 16 - 1,Zxx))
    imgZxx = lineMap(0,1,Zxx)
    imgZxx = imgZxx[::-1]
    # imgZxx = cv2.resize(imgZxx,(96,96))
    plt.figure()
    plt.imshow(imgZxx)
    plt.show()
    return imgZxx

# def testSTFT():
    
#     x0,n0 = getSig(t,F0,FP,FS,-38)
#     sig = n0 + x0
#     sig = getAcmSig(sig,100000,100000)
#     win = 10000
#     f, tt, Zxx = stft(sig, FS, nperseg=win,noverlap=win // 2)
#     Zxx = abs(Zxx)
#     df = FS / win
#     f = f[:int(FP / df) ]
#     Zxx = Zxx[:int(FP / df) ][:]
#     # imgZxx = np.uint16(lineMap(0,2 ** 16 - 1,Zxx))
#     imgZxx = lineMap(0,1,Zxx)
#     imgZxx = cv2.resize(imgZxx,(96,96))
#     plt.imshow(imgZxx)
#     plt.show()


def getSigEnhanced(sig,fS,f0):
    nT = int(fS // f0)
    hfT = int(nT // 2)
    sigHelp = sig
    for i in range(0,len(sig) - nT,nT):
        # plt.figure()
        sigHelp[i : i+hfT] = -sig[i+hfT:i+nT]
        sigHelp[i+hfT:i + nT] = -sig[i : i+hfT]
        # plt.plot(sigHelp[i:i+nT])
        # plt.show()
    return (sig + sigHelp) / 2
    

def getAcmSigEnhanced(sig,win,fS,f0):
    nT = int(fS // f0)
    nTW = int(win // nT)
    sigAcm = 0
    epoch = 2

    for _ in range(epoch):
        idx = np.arange(win)
        idxtmp = np.arange(win)
        order = np.arange(nTW)
        np.random.shuffle(order)
        for i in range(0,len(sig) - win,win):
            
            for idxo in range(len(order)):
                idx[idxo * nT:(idxo + 1) * nT] = idxtmp[order[idxo] * nT:(order[idxo] + 1)*nT]
            sigAcm = sigAcm + (sig[i:i+win])[idx]
    return sigAcm / epoch

            
def generateFakeSig(sig,fS,f0):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    order = np.arange(nW)
    np.random.shuffle(order)
    fakeSig = np.zeros(len(sig))
    for idxo in range(int(len(order))):
        if np.random.randint(50) % 2 == 0:
            fakeSig[idxo * nT : (idxo + 1) * nT] = sig[order[idxo] * nT : (order[idxo] + 1) * nT]
        else:
            fakeSig[idxo * nT : (idxo + 1) * nT] = -(sig[order[idxo] * nT : (order[idxo] + 1) * nT])[::-1]
    
    # nT = int(fS // f0)
    # nW = int(len(sig) // nT)
    # order = np.arange(nW * 4)
    # np.random.shuffle(order)
    # fakeSig = np.zeros(len(sig))
    # part4 = int(nT // 4)
    # for idxo in range(len(order)):
    #     od = order[idxo]
        
    #     idxo4 = idxo % 4
    #     od4 = od % 4
    #     if idxo4 == od4:
    #         sigPart4 = sig[od * part4 : (od + 1) * part4]
    #     elif (idxo4 == 0 and od4 == 1) or (idxo4 == 2 and od4 == 3) or (idxo4 == 1 and od4 == 0) or (idxo4 == 3 and od4 == 2):
    #         sigPart4 = (sig[od * part4 : (od + 1) * part4])[::-1]
    #     elif (idxo4 == 0 and od4 == 2) or (idxo4 == 1 and od4 == 3) or (idxo4 == 2 and od4 == 0) or (idxo4 == 3 and od4 == 1):
    #         sigPart4 = -sig[od * part4 : (od + 1) * part4]
    #     else:
    #         sigPart4 = -(sig[od * part4 : (od + 1) * part4])[::-1]
    #     fakeSig[idxo * part4 : (idxo + 1) * part4] = sigPart4

    return fakeSig

def testAcc(model,featureName,snr,fold = 5):
    filePath = "features/{}/{}.pkl".format(featureName,snr)
    if not os.path.exists(filePath):
        print("请先生成特征。")
        return
    with open(filePath,'rb') as f:
        X,y = pkl.load(f)
        if X.ndim == 1:
            X = X.reshape(X.shape[0],-1)
        print("X.shape:{}".format(X.shape))
        print("y.shape:{}".format(y.shape))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1 / fold,random_state = 3)
    
    model.fit(X_test,y_test)

    print("Acc:{:.4f}".format(model.score(X_test,y_test)))



def fakeAcm(sig,fS,f0,epoch):
    sigAcm = 0
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    orders = np.zeros((epoch,nT,nW),np.int)
    for i in range(epoch):
        for j in range(nT):
            orders[i][j] = np.arange(int(len(sig) / fS * f0))
            np.random.shuffle(orders[i][j])
    
    with open('order\\{}.pkl'.format(epoch),'rb') as f:
        orders = pkl.load(f)
    
    for e in range(epoch):
        # sigAcm = sigAcm + generateFakeSigV2(sig,fS,f0)
        sigAcm = sigAcm + generateFakeSigV4(sig,fS,f0,orders[e])
    with open('order\\{}.pkl'.format(epoch),'wb') as f:
        pkl.dump(orders,f)
    return sigAcm / epoch

def generateFakeSigV4(sig,fS,f0,order):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    for nt in range(nT):
        for nw in range(nW):
            fakeSig[nw * nT + nt] = sig[order[nt][nw] * nT + nt]
    return fakeSig

def generateFakeSigV3(sig,fS,f0,order):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    idx = np.arange(nW)
    for iT in range(nT):
        fakeSig[idx * nT + iT] = sig[order * nT + iT] 
    return fakeSig
def generateFakeSigV2(sig,fS,f0):
    nT = int(fS // f0)
    nW = int(len(sig) // nT)
    fakeSig = np.zeros(len(sig))
    idx = np.arange(nW)
    for iT in range(nT):
        order = np.arange(nW)
        np.random.shuffle(order)
        fakeSig[idx * nT + iT] = sig[order * nT + iT] 
    return fakeSig

def getAcmF(sig,w,s):
    fAcm = 0
    for i in range(0,len(sig) - w,s):
        fAcm += np.abs(np.fft.fft(sig[i:i+w]))
    return fAcm / len(range(0,len(sig) - w,s))


def getAcmOpt(n,epoch):
    fakeN = 0
    for _ in range(epoch):
        idx = np.arange(len(n))
        np.random.shuffle(idx)
        fakeN += n[idx]
    return fakeN / epoch

    


def showFeature(st,sp,se,featureName):
    for snr in range(st,sp,se):
        with open("features\\test\\{}{}.pkl".format(snr,featureName),'rb') as f:
            X,y = pkl.load(f)
        X = X[:,-1]
        plt.figure()
        for i in range(len(y)):
            if y[i] == 0:
                plt.scatter(X[i],X[i],c = 'r')
            elif y[i] == 1:
                plt.scatter(X[i],X[i],c = 'b')
        plt.show()



def getSinglePointSig(sig,fS,f0):
    df = fS / len(sig)
    f = np.arange(len(sig)) / len(sig) * fS
    f02N = np.where(np.abs(f - f0) < df / 2)[0][0]
    sigF = np.fft.fft(sig)
    singlePointSigF = np.zeros(len(sigF),dtype=complex)
    singlePointSigF[f02N] = sigF[f02N]
    singlePointSigF[-f02N] = sigF[-f02N]

    singlePointSig = np.real(np.fft.ifft(singlePointSigF))
    return singlePointSig,singlePointSigF[f02N]


