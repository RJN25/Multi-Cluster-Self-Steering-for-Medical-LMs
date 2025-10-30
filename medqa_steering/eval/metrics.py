import numpy as np
from sklearn.metrics import roc_auc_score

def brier_multiclass(probs, y):
    # probs: [N,4], y:[N] in 0..3
    N, K = probs.shape
    onehot = np.eye(K)[y]
    return np.mean(np.sum((probs - onehot)**2, axis=1))

def ece_multiclass(probs, y, n_bins=15):
    # Expected Calibration Error (one-vs-rest, averaged)
    N,K = probs.shape
    eces = []
    for k in range(K):
        conf = probs[:,k]
        lab = (y==k).astype(int)
        bins = np.linspace(0,1,n_bins+1)
        accs, confs, weights = [], [], []
        for i in range(n_bins):
            idx = (conf>=bins[i]) & (conf<bins[i+1])
            if idx.sum()==0: continue
            accs.append(lab[idx].mean()); confs.append(conf[idx].mean()); weights.append(idx.mean())
        if len(weights)==0: continue
        eces.append(np.sum(np.array(weights)*np.abs(np.array(accs)-np.array(confs))))
    return float(np.mean(eces)) if eces else 0.0

def macro_auroc_ovr(probs, y):
    # safe AUROC even if a class missing; ignore those classes
    K = probs.shape[1]
    aucs=[]
    for k in range(K):
        yk = (y==k).astype(int)
        if yk.sum()==0 or yk.sum()==len(yk): continue
        aucs.append(roc_auc_score(yk, probs[:,k]))
    return float(np.mean(aucs)) if aucs else float('nan')
