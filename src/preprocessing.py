from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def load_data():
    d = load_breast_cancer(as_frame=True)
    return d.data, (d.target == 0).astype(int).rename('malignant')

def split_data():
    X,y = load_data()
    a,c,ya,yc = train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    a,b,ya,yb = train_test_split(a,ya,test_size=.25,stratify=ya,random_state=SEED)
    return a,b,c,ya,yb,yc

def pipeline(model):
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),model)
