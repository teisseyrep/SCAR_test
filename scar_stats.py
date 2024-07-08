import numpy as np
from scipy import stats
from sklearn import  metrics
from sklearn.naive_bayes import GaussianNB


def kl_stat(X,s_tilde,y_tilde):
    p = X.shape[1]
    w1 = np.where(s_tilde==1)[0]
    w2 = np.where(y_tilde==1)[0]
    X1 = X[w1,:]
    X2 = X[w2,:]
    u1 = np.mean(X1,axis=0)
    u2 = np.mean(X2,axis=0)
    a = np.var(X1,axis=0)
    b = np.var(X2,axis=0)
    
    term1 = np.dot((u2-u1)**2,1/b)
    term2 = np.sum(a/b)
    det1 = np.prod(a)
    det2 = np.prod(b)
    term3 = np.log(det1/det2)
    
    res = 0.5*(term1+term2-term3 -p)
    return res

def kl_stat_cov(X,s_tilde,y_tilde):
    p = X.shape[1]
    w1 = np.where(s_tilde==1)[0]
    w2 = np.where(y_tilde==1)[0]
    X1 = X[w1,:]
    X2 = X[w2,:]
    u1 = np.mean(X1,axis=0)
    u2 = np.mean(X2,axis=0)
    r = u2-u1
      
    Sigma1 = np.cov(X1,rowvar=False) + np.diag(0.001 * np.ones(p)) 
    Sigma2 = np.cov(X2,rowvar=False) + np.diag(0.001 * np.ones(p)) 
    det1 = np.linalg.det(Sigma1)
    det2 = np.linalg.det(Sigma2)
    Sigma2_inv = np.linalg.inv(Sigma2) 
    term1 = np.dot(r.T, np.dot(Sigma2_inv,r))
    term2 = np.sum(np.diag(np.dot(Sigma2_inv,Sigma1)))
    term3 = np.log(det1/det2)
    
    res = 0.5*(term1+term2-term3 -p)
    return res

def ks_stat(X,s_tilde,y_tilde):
   
    p = X.shape[1]
    w1 = np.where(s_tilde==1)[0]
    w2 = np.where(y_tilde==1)[0]
    X1 = X[w1,:]
    X2 = X[w2,:]
    ks_vec = np.zeros(p)
    for j in np.arange(p):
        sample1 = X1[:,j]
        sample2 = X2[:,j]
        ks = stats.kstest(sample1, sample2, mode='asymp')
        ks_vec[j] = ks.statistic
    
    res = np.sum(ks_vec)
    return res

def nb_stat(X,s_tilde,y_tilde):
    w1 = np.where(y_tilde==1)[0]
    w0 = np.where(s_tilde==1)[0]
    X_temp1 = X[w1,:]
    X_temp0 = X[w0,:]
    X_temp = np.concatenate([X_temp1,X_temp0])
    class_temp = np.append(np.ones(w1.shape[0]),np.zeros(w0.shape[0]))
    gnb = GaussianNB()
    class_pred = gnb.fit(X_temp, class_temp).predict_proba(X_temp)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(class_temp, class_pred, pos_label=1)
    res = metrics.auc(fpr, tpr)-0.5
    
    return res



