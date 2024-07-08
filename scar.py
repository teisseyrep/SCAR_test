import numpy as np
from scar_stats import kl_stat, kl_stat_cov, ks_stat, nb_stat


def find_unlabeled_positive(s,sx,hat_c):
    n = s.shape[0]
    hat_s = np.mean(s)
    hat_y = hat_s/hat_c
    ile = int(n*(hat_y-hat_s))
    if ile<10:
        ile = 10
    w0 =  np.where(s==0)[0] 
    sx0 = sx[w0]
    sel0 = np.argsort(-sx0)[0:ile]
    sel1 = w0[sel0]
    return sel1

def generate_scar_s_sx(s,sx,hat_c):

    n = s.shape[0]
    hat_s = np.mean(s)
    hat_y = hat_s/hat_c
    ile = int(n*(hat_y-hat_s))
    if ile<10:
        ile = 10
    w0 =  np.where(s==0)[0] 
    w1 =  np.where(s==1)[0]
    sx0 = sx[w0]
    sel0 = np.argsort(-sx0)[0:ile]
    sel1 = w0[sel0]
    sel = np.union1d(sel1,w1)
    y_tilde = np.zeros(n)
    y_tilde[sel]=1   
    s_tilde = np.zeros(n)
    for i in np.arange(0,n,1):
        if y_tilde[i]==1:
              s_tilde[i]=np.random.binomial(1, hat_c, size=1)
    return s_tilde, y_tilde   



def make_scar_test(X,s,hat_c,clf,B=500,alpha=0.05,stat="kl"):

       n = X.shape[0]    

       model_naive = clf
       model_naive.fit(X,s)
       sx = model_naive.predict_proba(X)[:,1]
     
       Tstat = np.zeros(B)

       sel_up = find_unlabeled_positive(s,sx,hat_c)
       y_up = np.zeros(n)
       y_up[sel_up] = 1

       for iter in np.arange(B):
    
         
            s_tilde, y_tilde = generate_scar_s_sx(s,sx,hat_c)

                
            if stat=="kl":
                Tstat[iter] = kl_stat(X,s_tilde,y_tilde)
            elif stat=="ks":
                Tstat[iter] = ks_stat(X,s_tilde,y_tilde) 
            elif stat=="nb":
                Tstat[iter] = nb_stat(X,s_tilde,y_tilde)                 
            elif stat=="klcov":
                Tstat[iter] = kl_stat_cov(X,s_tilde,y_tilde)    

            else:
                raise Exception("stat NOT found!")
                

       if stat=="kl":
            Tstat0 = kl_stat(X,s,y_tilde)
            pv = np.where(Tstat>Tstat0)[0].shape[0]/B
       elif stat=="ks":
            Tstat0 = ks_stat(X,s,y_tilde)
            pv = np.where(Tstat>Tstat0)[0].shape[0]/B  
       elif stat=="nb":
            Tstat0 = nb_stat(X,s,y_tilde)
            pv = np.where(Tstat>Tstat0)[0].shape[0]/B             
       elif stat=="klcov":
            Tstat0 = kl_stat_cov(X,s,y_tilde)
            pv = np.where(Tstat>Tstat0)[0].shape[0]/B                       
       else:
            raise Exception("stat NOT found!")     
        
       if pv<alpha:
           reject = 1
       else:
            reject = 0
       return reject, pv, Tstat, Tstat0




