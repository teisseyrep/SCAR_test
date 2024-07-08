import numpy as np


def generate_artificial_data_discr(prior=0.5,n=2000,p=10,b=1,xdistr="norm",Sigma0=0,Sigma1=0):

    
    y = np.zeros(n)
    ytest = np.zeros(n)
    for i in np.arange(0,n,1):
        y[i] = np.random.binomial(1, prior, size=1)
        ytest[i] = np.random.binomial(1, prior, size=1)

    
    X=np.zeros((n,p))
    ind1 = np.where(y==1)[0]
    ind0 = np.where(y==0)[0]
    n1 = ind1.shape[0]
    n0 = ind0.shape[0]
    
    Xtest=np.zeros((n,p))
    ind1test = np.where(ytest==1)[0]
    ind0test = np.where(ytest==0)[0]
    n1test = ind1test.shape[0]
    n0test = ind0test.shape[0]
    
    mean0 = np.zeros(p)
    mean1 = b*np.ones(p)
    
    if xdistr=='norm':
        for j in np.arange(0,p,1):
                X[ind0,:]=np.random.multivariate_normal(mean0,Sigma0,size=n0)
                X[ind1,:]=np.random.multivariate_normal(mean1,Sigma1,size=n1)
                Xtest[ind0test,:]=np.random.multivariate_normal(mean0,Sigma0,size=n0test)
                Xtest[ind1test,:]=np.random.multivariate_normal(mean1,Sigma1,size=n1test)
    else:
          print('Argument xdistr is not defined')      
 
        
    
    return y,ytest,X,Xtest   


 