from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
import numpy as np

def evaluate(y,p,t=.5):
    pred = p >= t
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),brier=brier_score_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def threshold(y,p):
    grid = np.linspace(.05,.95,91)
    return float(grid[np.argmax([f1_score(y,p>=t,zero_division=0) for t in grid])])
