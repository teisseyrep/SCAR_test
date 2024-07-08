import numpy as np
import pandas as pd

from labelling import make_pu_labels
from utils import  make_binary_class, remove_collinear
from scar import make_scar_test
from sklearn.ensemble import RandomForestClassifier


path = 'data/'
ds = 'Breast-w'


c = 0.5
g = 5
label_scheme = "S1"
stat = 'ks'
B=200
clf = RandomForestClassifier()    


                    
# Load and prepare data:
df_name = path + ds + '.csv'
df = pd.read_csv(df_name, sep=',')
if np.where(df.columns.values=="BinClass")[0].shape[0]>0:
    del df['BinClass']
df = df.to_numpy()
p = df.shape[1]-1
y = df[:,p]
y = make_binary_class(y)
X = df[:,0:p]
selected_columns = remove_collinear(X,0.95)
X = X[:,selected_columns]
hat_y = np.mean(y) 

                
                
#Create PU dataset:
s, ex_true, a = make_pu_labels(X,y,label_scheme=label_scheme,c=c,g=g)



#Run test:
hat_s = np.mean(s)
hat_c = hat_s/hat_y

reject, pv, Tstat, Tstat0 = make_scar_test(X,s,hat_c,clf,B=B,alpha=0.05,stat=stat)
          
print("H0 rejected (yes=1, np = 0): ", reject,'\n')
print("p-value: ", pv,'\n')
print("Observed test statistic: ",Tstat0,'\n')
print("95% quantile of the generated null distribution of the test statistic:",np.quantile(Tstat,0.95),'\n')
print("Generated null distribution of the test statistic: " ,Tstat,'\n')



          



