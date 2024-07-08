import numpy as np
from labelling import make_pu_labels
from scar import make_scar_test
from sklearn.ensemble import RandomForestClassifier
from artificial import  generate_artificial_data_discr


path = 'data/'

c = 0.5
stat_seq = ['kl','klcov','ks','nb']
p = 10
nsym = 100
label_scheme = "S1"
b = 1
n_seq = np.arange(100,1000,50)
B = 300
g_seq = [0,1,2]
clf = RandomForestClassifier()   



for stat in stat_seq:

    for ds in ['art1','art2']:
    #for ds in ['art1']:
        print('\n ****** DATASET=:', ds,"******")
        res = np.zeros((len(n_seq),len(g_seq)))
        
        power = np.zeros(nsym)
        
        i=0
        for n in n_seq:
            print('\n ===n=:', n,"===")
            j=0
            for g in g_seq:
                print('\n---g:', g,"---")
                for sym in np.arange(nsym):
                    
                    print("|",end='')
                        
                    if ds=="art1":
                        Sigma0 = np.diag(np.ones(p))
                        Sigma1 = np.diag(np.ones(p))
                        y,ytest,X,Xtest = generate_artificial_data_discr(prior=0.5,n=n,p=p,b=b,xdistr="norm",Sigma0=Sigma0,Sigma1=Sigma1)
                        
                    if ds=="art2":
                        Sigma0 = np.diag(np.ones(p))
                        rho=0.5
                        Sigma1 = np.zeros((p,p))
                        for i1 in np.arange(p):
                            for j1 in np.arange(p):
                               Sigma1[i1,j1] = np.power(rho,np.abs(i1-j1))
                    
                        y,ytest,X,Xtest = generate_artificial_data_discr(prior=0.5,n=n,p=p,b=b,xdistr="norm",Sigma0=Sigma0,Sigma1=Sigma1)
                        
                        
                    #Create PU dataset:
                    s, ex_true, a = make_pu_labels(X,y,label_scheme=label_scheme,c=c,g=g)
                
                    hat_y = np.mean(y)
                    hat_s = np.mean(s)
                    hat_c = hat_s/hat_y
            
            
                    reject, pv, Tstat, Tstat0 = make_scar_test(X,s,hat_c,clf,B=B,alpha=0.05,stat=stat)
                    
                    power[sym] = reject
                        
                res[i,j] = np.mean(power)   
                j=j+1
            i=i+1
        
        file_out = 'results/results_'  + 'ds' +ds + 'c' + str(c) + 'p' + str(p) + 'label_scheme' + label_scheme + 'stat' + str(stat) + ".txt"  
        np.savetxt(file_out, res,fmt='%1.3f')
