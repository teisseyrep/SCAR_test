import numpy as np
import pandas as pd
from labelling import make_pu_labels
from utils import  make_binary_class, mi_filter, remove_collinear
from scar import make_scar_test
from sklearn.ensemble import RandomForestClassifier


path = 'data/'
ds_seq =['Breast-w','Wdbc','Banknote-authentication','Segment','CIFAR10_sample','USPS_sample','Fashion_sample']
c = 0.5
stat_seq = ['kl','klcov','ks','nb']
nsym = 100
g_seq = [0,1,2]
label_scheme = "S1"
B = 300
clf = RandomForestClassifier()    



for ds in ds_seq:
    print('\n ****** DATASET=:', ds,"******")
    res = np.zeros((len(stat_seq), len(g_seq)))
    
    power = np.zeros(nsym)
    
    i=0
    for stat in stat_seq:
        print('\n ===stat=:', stat,"===")
        j=0
        for g in g_seq:
            print('\n---label_scheme:', label_scheme,"---")
            for sym in np.arange(nsym):
                
                print("|",end='')
                    
                # Load data:
                df_name = path + ds + '.csv'
                df = pd.read_csv(df_name, sep=',')
                if np.where(df.columns.values=="BinClass")[0].shape[0]>0:
                    del df['BinClass']
                df = df.to_numpy()
                p = df.shape[1]-1
                yall = df[:,p]
                yall = make_binary_class(yall)
                Xall = df[:,0:p]
                selected_columns = remove_collinear(Xall,0.95)
                Xall = Xall[:,selected_columns]
                sel = mi_filter(Xall,yall,pmax=20)
                Xall = Xall[:,sel]
                X = Xall
                y = yall
                
                stds = np.std(X,axis=0)
                del0  = np.where(stds==0)[0]
                if del0.shape[0]>0:
                    X=np.delete(X,del0,axis=1)
                
                
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
        
        file_out = 'results/results_'  + 'ds' +ds + 'c' + 'label_scheme' + label_scheme + 'stat' + str(stat) + ".txt"  
        np.savetxt(file_out, res,fmt='%1.3f')

