import numpy as np
import warnings   
from utils import sigma
from sklearn.linear_model import LogisticRegression

def make_pu_labels(X,y,label_scheme='S1',c=0.5,g=1):
    
    
    n = X.shape[0]
    p = X.shape[1]
    s = np.zeros(n)
    ex_true = np.zeros(n)    
    a_opt = c
    
    if g==0:
        label_scheme="S0"
    
    if label_scheme!='S0':
        model_oracle = LogisticRegression()
        model_oracle.fit(X,y)
        prob_true=model_oracle.predict_proba(X)[:,1]
        prob_true[np.where(prob_true>0.999)[0]] = 0.999
        prob_true[np.where(prob_true<0.001)[0]] = 0.001
        
    if label_scheme=='S0':
        for i in np.arange(0,n,1):
            ex_true[i] = c         
    elif label_scheme=='S1':
        jsel = 0
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
           for i in np.arange(0,n,1):
               ex_true[i] = sigma(g*X[i,jsel] + a)
           score[k] = np.abs(np.mean(ex_true[w1])-c)
           k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
           ex_true[i] = sigma(g*X[i,jsel] + a_opt)                  
    elif label_scheme=='S2':
        
        lin_pred = np.log(prob_true/(1-prob_true))  
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = sigma(g*lin_pred[i]/np.sqrt(p) + a)
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = sigma(g*lin_pred[i]/np.sqrt(p) + a_opt)                 
    elif label_scheme=='S3':           
        lin_pred = np.log(prob_true/(1-prob_true))  
        
        a_seq = np.linspace(start=-10, stop=10, num=100)
        score = np.zeros(100)
        k=0
        w1 = np.where(y==1)[0]
        for a in a_seq:
            for i in np.arange(0,n,1):
                ex_true[i] = sigma(g*lin_pred[i]/np.sqrt(p) + a)**10
            score[k] = np.abs(np.mean(ex_true[w1])-c)
            k=k+1
        a_opt = a_seq[np.argmin(score)]
        for i in np.arange(0,n,1):
            ex_true[i] = sigma(g*lin_pred[i]/np.sqrt(p) + a_opt)**10                     
    else:
        print('Argument label_scheme is not defined')
                    
    for i in np.arange(0,n,1):
        if y[i]==1:
            s[i]=np.random.binomial(1, ex_true[i], size=1)
        
    if np.sum(s)<=1:
        s[np.random.choice(s.shape[0],2,replace=False)]=1
        warnings.warn('Warning: <2 observations with s=1. Two random instances were assigned label s=1.')
        
    return s, ex_true, a_opt



def label_transform_MNIST(y):
    w1 = (y%2==0)
    w0 = (y%2!=0)
    y[w1]=1
    y[w0]=0
    return y

def label_transform_CIFAR10(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 8, 9]:
            y[i]=1
        else:
            y[i]=0
    return y
   
def label_transform_USPS(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 2, 3, 4]:
            y[i]=1
        else:
            y[i]=0
    return y
 
def label_transform_Fashion(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 2, 3, 4, 6]:
            y[i]=1
        else:
            y[i]=0
    return y       
    



