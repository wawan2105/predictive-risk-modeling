import json
import platform
from importlib.metadata import version
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from .preprocessing import ROOT, SEED, load_data, split_data, pipeline
from .evaluate import evaluate, threshold
plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})

def save(name):
    plt.tight_layout()
    plt.savefig(ROOT / 'outputs/figures' / name,bbox_inches='tight')
    plt.close()

def eda():
    X,y = load_data()
    X.assign(malignant=y).to_csv(ROOT/'data/raw/breast_cancer.csv',index_label='sample_id')
    X.describe().T.to_csv(ROOT/'outputs/tables/descriptive_statistics.csv')
    audit = dict(rows=len(X),features=X.shape[1],missing=int(X.isna().sum().sum()),duplicates=int(X.duplicated().sum()),malignant=int(y.sum()))
    (ROOT/'outputs/data_audit.json').write_text(json.dumps(audit,indent=2))
    fig,ax = plt.subplots(figsize=(6,4))
    y.value_counts().sort_index().rename(index={0:'Benign',1:'Malignant'}).plot.bar(ax=ax,rot=0,color=['#298c8c','#cf704d'])
    ax.set(title='Dataset class distribution',ylabel='Samples',xlabel='Diagnosis')
    save('class_distribution.png')
    fig,ax = plt.subplots(figsize=(9,7))
    corr = X.corr()
    im = ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=6)
    ax.set_yticks(range(len(corr)),corr.columns,fontsize=6)
    fig.colorbar(im,ax=ax,label='Pearson correlation')
    ax.set_title('Feature correlations | descriptive only')
    save('feature_correlations.png')
    return audit

def run():
    audit = eda()
    a,b,c,ya,yb,yc = split_data()
    split = []
    for name,X,y in [('train',a,ya),('validation',b,yb),('test',c,yc)]:
        X.assign(malignant=y).to_csv(ROOT/f'data/processed/{name}.csv',index_label='sample_id')
        split.append(dict(split=name,rows=len(X),malignant=int(y.sum())))
    pd.DataFrame(split).to_csv(ROOT/'outputs/tables/split_summary.csv',index=False)
    models = {'Logistic Regression':LogisticRegression(max_iter=3000,random_state=SEED),'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,random_state=SEED),'Random Forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=1),'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=SEED)}
    rows, fitted = [],{}
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for name,model in models.items():
        pipe = pipeline(model)
        s = cross_validate(pipe,a,ya,cv=cv,scoring={'auc':'roc_auc','f1':'f1','recall':'recall'},n_jobs=1)
        pipe.fit(a,ya)
        fitted[name] = pipe
        row = dict(model=name,cv_auc_mean=s['test_auc'].mean(),cv_auc_sd=s['test_auc'].std(ddof=1),cv_f1_mean=s['test_f1'].mean(),cv_recall_mean=s['test_recall'].mean())
        row.update({'validation_'+k:v for k,v in evaluate(yb,pipe.predict_proba(b)[:,1]).items()})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('cv_auc_mean',ascending=False)
    table.to_csv(ROOT/'outputs/tables/model_comparison.csv',index=False)
    name = table.iloc[0]['model']
    best = fitted[name]
    t = threshold(yb,best.predict_proba(b)[:,1])
    # Keep the training fit to preserve the validation-selected threshold.
    p = best.predict_proba(c)[:,1]
    result = evaluate(yc,p,t)
    result.update(model=name,threshold=t,seed=SEED)
    (ROOT/'outputs/test_metrics.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame([result]).to_csv(ROOT/'outputs/tables/test_metrics.csv',index=False)
    pd.DataFrame(dict(sample_id=c.index,actual=yc.to_numpy(),probability=p,predicted=(p>=t).astype(int))).to_csv(ROOT/'outputs/tables/test_predictions.csv',index=False)
    joblib.dump(dict(pipeline=best,threshold=t,features=list(a.columns)),ROOT/'outputs/model.joblib')
    fig,ax = plt.subplots(figsize=(6,5))
    RocCurveDisplay.from_predictions(yc,p,ax=ax,name=name)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_title('Held-out test ROC curve')
    save('test_roc.png')
    fig,ax = plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(yc,p>=t,ax=ax,display_labels=['Benign','Malignant'],cmap='Blues',colorbar=False)
    ax.set_title(f'Test confusion matrix | threshold={t:.2f}')
    save('confusion_matrix.png')
    fig,ax = plt.subplots(figsize=(6,5))
    CalibrationDisplay.from_predictions(yc,p,n_bins=6,strategy='quantile',ax=ax)
    ax.set_title('Test calibration | small-sample diagnostic')
    save('calibration.png')
    perm = permutation_importance(best,b,yb,scoring='roc_auc',n_repeats=20,random_state=SEED,n_jobs=1)
    imp = pd.DataFrame(dict(feature=b.columns,mean_auc_decrease=perm.importances_mean,sd_auc_decrease=perm.importances_std)).sort_values('mean_auc_decrease',ascending=False)
    imp.to_csv(ROOT/'outputs/tables/permutation_importance.csv',index=False)
    top = imp.head(10).iloc[::-1]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.barh(top.feature,top.mean_auc_decrease,xerr=top.sd_auc_decrease,color='#298c8c')
    ax.set(xlabel='Validation ROC-AUC decrease (mean +/- SD)',title='Permutation importance | 20 repeats')
    save('permutation_importance.png')
    fig,ax = plt.subplots(figsize=(9,4))
    PartialDependenceDisplay.from_estimator(best,b,features=imp.feature.head(2).tolist(),ax=ax)
    fig.suptitle('Partial dependence | correlated predictors limit interpretation')
    save('partial_dependence.png')
    env = {'python':platform.python_version(),**{v:version(v) for v in ['numpy','pandas','scikit-learn','matplotlib','joblib','nbformat']}}
    (ROOT/'outputs/environment.json').write_text(json.dumps(env,indent=2))
    (ROOT/'requirements-lock.txt').write_text(chr(10).join(f'{k}=={v}' for k,v in env.items() if k!='python')+chr(10))
    summary = ['# Executive summary','Author: Wawan Saputra','Educational Wisconsin Diagnostic Breast Cancer classification study. Positive class: malignant.',
        f'Split train/validation/test: {len(a)}/{len(b)}/{len(c)}. Selected by training-only five-fold ROC-AUC: {name}.',
        f'Validation-selected F1 threshold: {t:.2f}. Test ROC-AUC: {result["roc_auc"]:.4f}; precision: {result["precision"]:.4f}; recall: {result["recall"]:.4f}; F1: {result["f1"]:.4f}.',
        f'Test false negatives: {result["fn"]}; false positives: {result["fp"]}; Brier score: {result["brier"]:.4f}.',
        '## Interpretation and recommendations','Compare error costs before choosing a threshold. F1 optimization does not encode clinical costs. Importance describes model reliance, not causation. Correlated features share importance and can make partial dependence unrealistic.',
        '## Limitations and Future Work','Small historical single-source sample, no external validation, uncertainty intervals, patient grouping identifiers or subgroup fairness audit. Image-derived classification does not establish population risk or screening utility. This educational model is not validated for medical decisions. Next: external validation, repeated/nested evaluation, uncertainty estimates and domain-informed cost-sensitive thresholds.',
        '## CV-ready bullets',f'- Compared four classification pipelines using stratified five-fold cross-validation and a held-out test set of {len(c)} samples.',f'- Achieved test ROC-AUC {result["roc_auc"]:.3f} and F1 {result["f1"]:.3f} with {name}; assessed calibration and interpreted model reliance with permutation importance.']
    (ROOT/'outputs/executive_summary.md').write_text((chr(10)*2).join(summary)+chr(10),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__':
    run()
