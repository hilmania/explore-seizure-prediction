from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def build_models(random_state: int = 42, class_weight='balanced') -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    models['logreg'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state))
    ])
    models['svm_rbf'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, class_weight=class_weight, random_state=random_state))
    ])
    models['rf'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=None, class_weight=class_weight, random_state=random_state, n_jobs=-1))
    ])
    models['gb'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(random_state=random_state))
    ])
    models['knn'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7))
    ])
    return models
