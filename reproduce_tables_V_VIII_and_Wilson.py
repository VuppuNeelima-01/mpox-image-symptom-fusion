import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

p='per_sample_predictions.csv'
df=pd.read_csv(p)
models={'Image-only':'image_only_pred','Symptom-only':'symptom_only_pred','Fusion':'fusion_pred'}
classes=sorted(df.true_label.unique())
rows=[]
for name,col in models.items():
    acc=accuracy_score(df.true_label,df[col])
    pr,re,f1,_=precision_recall_fscore_support(df.true_label,df[col],labels=classes,zero_division=0)
    rows.append([name,acc,pr.mean(),re.mean(),f1.mean()])
print(pd.DataFrame(rows,columns=['Model','Accuracy','Macro Precision','Macro Recall','Macro F1']))
for name,col in models.items():
    pr,re,f1,sup=precision_recall_fscore_support(df.true_label,df[col],labels=classes,zero_division=0)
    print('\
',name)
    print(pd.DataFrame({'Class':classes,'Precision':pr,'Recall':re,'F1':f1,'Support':sup}))
    cm=confusion_matrix(df.true_label,df[col],labels=classes)
    print('Confusion matrix\
',cm)
    k=int((df[col]==df.true_label).sum())
    lo,hi=proportion_confint(k,len(df),alpha=0.05,method='wilson')
    print('Wilson 95%:',k,len(df),k/len(df),lo,hi)
