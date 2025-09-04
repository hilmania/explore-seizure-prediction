import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, chi2, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureSelector:
    """
    Advanced feature selection methods for EEG seizure prediction
    Combines multiple selection strategies to find optimal feature combinations
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.imputer = SimpleImputer(strategy='median')

    def _preprocess_features(self, X: np.ndarray, fit_imputer: bool = True) -> np.ndarray:
        """
        Preprocess features by handling NaN values and infinite values

        Args:
            X: Feature matrix that may contain NaN/inf values
            fit_imputer: Whether to fit the imputer (True for training, False for transform)

        Returns:
            Cleaned feature matrix
        """
        # Convert to float to handle any data type issues
        X = np.asarray(X, dtype=np.float64)

        # Replace infinite values with NaN first
        X = np.where(np.isfinite(X), X, np.nan)

        # Check if there are any NaN values
        if np.any(np.isnan(X)):
            print(f"[feature-selection] Found {np.sum(np.isnan(X))} NaN values, imputing with median...")

            if fit_imputer:
                X = self.imputer.fit_transform(X)
            else:
                X = self.imputer.transform(X)

        # Final check for any remaining problematic values
        if not np.all(np.isfinite(X)):
            print("[feature-selection] Warning: Some non-finite values remain, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def univariate_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        method: str = 'f_classif',
        k: int = 20
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Univariate feature selection using statistical tests

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: Names of features
            method: 'f_classif', 'chi2', or 'mutual_info'
            k: Number of features to select

        Returns:
            Dict with selected indices, names, and scores
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        # Choose scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'chi2':
            # Ensure non-negative features for chi2
            X = np.abs(X)
            score_func = chi2
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")

        # Select features
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get results
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        scores = selector.scores_

        result = {
            'method': f'univariate_{method}',
            'selected_indices': selected_indices,
            'selected_names': selected_names,
            'scores': scores,
            'X_selected': X_selected
        }

        self.feature_scores_[f'univariate_{method}'] = scores
        self.selected_features_[f'univariate_{method}'] = selected_indices

        return result

    def recursive_feature_elimination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        estimator_type: str = 'rf',
        n_features: int = 20,
        cv_folds: int = 3,
        use_cv: bool = True
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Recursive Feature Elimination with optional cross-validation

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            estimator_type: 'rf', 'svm', or 'logreg'
            n_features: Number of features to select
            cv_folds: Number of CV folds for RFECV
            use_cv: Whether to use RFECV or RFE

        Returns:
            Dict with selected features and rankings
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        # Choose estimator
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif estimator_type == 'svm':
            estimator = SVC(
                kernel='linear',
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif estimator_type == 'logreg':
            estimator = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator_type}")

        # Perform RFE
        if use_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring='roc_auc',
                min_features_to_select=min(n_features, X.shape[1])
            )
        else:
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=1
            )

        X_selected = selector.fit_transform(X, y)

        # Get results
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        rankings = selector.ranking_

        result = {
            'method': f'rfe_{estimator_type}{"_cv" if use_cv else ""}',
            'selected_indices': selected_indices,
            'selected_names': selected_names,
            'rankings': rankings,
            'X_selected': X_selected,
            'n_features_selected': len(selected_indices)
        }

        if use_cv and hasattr(selector, 'cv_results_'):
            result['cv_scores'] = selector.cv_results_

        self.selected_features_[result['method']] = selected_indices

        return result

    def model_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        estimator_type: str = 'rf',
        threshold: str = 'median'
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Model-based feature selection using feature importance

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            estimator_type: 'rf', 'logreg'
            threshold: Threshold for importance ('median', 'mean', or float)

        Returns:
            Dict with selected features and importance scores
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        # Choose estimator
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif estimator_type == 'logreg':
            estimator = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000,
                penalty='l1',
                solver='liblinear'
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator_type}")

        # Select features
        selector = SelectFromModel(
            estimator=estimator,
            threshold=threshold
        )
        X_selected = selector.fit_transform(X, y)

        # Get results
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]

        # Get feature importance
        estimator.fit(X, y)
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_[0])
        else:
            importances = np.ones(X.shape[1])

        result = {
            'method': f'model_{estimator_type}',
            'selected_indices': selected_indices,
            'selected_names': selected_names,
            'importances': importances,
            'X_selected': X_selected,
            'threshold_used': selector.threshold_
        }

        self.feature_scores_[result['method']] = importances
        self.selected_features_[result['method']] = selected_indices

        return result

    def variance_based_selection(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        threshold: float = 0.0
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Remove features with low variance

        Args:
            X: Feature matrix
            feature_names: Names of features
            threshold: Variance threshold

        Returns:
            Dict with selected features
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)

        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]
        variances = selector.variances_

        result = {
            'method': 'variance_threshold',
            'selected_indices': selected_indices,
            'selected_names': selected_names,
            'variances': variances,
            'X_selected': X_selected,
            'threshold': threshold
        }

        self.selected_features_['variance_threshold'] = selected_indices

        return result

    def sequential_forward_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        estimator_type: str = 'rf',
        max_features: int = 20,
        scoring: str = 'roc_auc',
        cv_folds: int = 3
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Sequential Forward Selection

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            estimator_type: Type of estimator
            max_features: Maximum number of features to select
            scoring: Scoring metric
            cv_folds: Number of CV folds

        Returns:
            Dict with selected features and scores
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        # Choose estimator
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif estimator_type == 'svm':
            estimator = SVC(
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif estimator_type == 'logreg':
            estimator = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator_type}")

        # Setup CV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scorer = get_scorer(scoring)

        # Sequential forward selection
        selected_features = []
        remaining_features = list(range(X.shape[1]))
        scores_history = []

        for i in range(min(max_features, X.shape[1])):
            best_score = -np.inf
            best_feature = None

            for feature in remaining_features:
                current_features = selected_features + [feature]
                X_subset = X[:, current_features]

                # Cross-validation score
                cv_scores = cross_val_score(
                    clone(estimator), X_subset, y,
                    cv=cv, scoring=scoring
                )
                mean_score = np.mean(cv_scores)

                if mean_score > best_score:
                    best_score = mean_score
                    best_feature = feature

            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                scores_history.append(best_score)
            else:
                break

        selected_names = [feature_names[i] for i in selected_features]

        result = {
            'method': f'sfs_{estimator_type}',
            'selected_indices': np.array(selected_features),
            'selected_names': selected_names,
            'scores_history': scores_history,
            'X_selected': X[:, selected_features] if selected_features else X[:, :0]
        }

        self.selected_features_[result['method']] = np.array(selected_features)

        return result

    def ensemble_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        methods: List[str] = None,
        top_k: int = 20,
        voting_threshold: float = 0.5
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Ensemble feature selection combining multiple methods

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            methods: List of methods to combine
            top_k: Number of top features to select
            voting_threshold: Threshold for ensemble voting

        Returns:
            Dict with ensemble selected features
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Preprocess features to handle NaN values
        X = self._preprocess_features(X, fit_imputer=True)

        if methods is None:
            methods = ['univariate_f_classif', 'rfe_rf', 'model_rf']

        # Apply each method
        method_results = {}
        for method in methods:
            if method.startswith('univariate'):
                _, score_func = method.split('_', 1)
                result = self.univariate_selection(X, y, feature_names, score_func, top_k)
            elif method.startswith('rfe'):
                _, estimator = method.split('_', 1)
                result = self.recursive_feature_elimination(X, y, feature_names, estimator, top_k)
            elif method.startswith('model'):
                _, estimator = method.split('_', 1)
                result = self.model_based_selection(X, y, feature_names, estimator)
            else:
                continue

            method_results[method] = result

        # Ensemble voting
        n_features = X.shape[1]
        feature_votes = np.zeros(n_features)

        for method, result in method_results.items():
            selected_indices = result['selected_indices']
            feature_votes[selected_indices] += 1

        # Normalize votes
        feature_votes = feature_votes / len(method_results)

        # Select features based on voting threshold
        ensemble_selected = np.where(feature_votes >= voting_threshold)[0]

        # If too few features, select top-k by votes
        if len(ensemble_selected) < top_k:
            ensemble_selected = np.argsort(feature_votes)[-top_k:]

        ensemble_names = [feature_names[i] for i in ensemble_selected]

        result = {
            'method': 'ensemble',
            'selected_indices': ensemble_selected,
            'selected_names': ensemble_names,
            'feature_votes': feature_votes,
            'X_selected': X[:, ensemble_selected],
            'method_results': method_results,
            'voting_threshold': voting_threshold
        }

        self.selected_features_['ensemble'] = ensemble_selected

        return result

    def get_feature_importance_summary(self) -> Dict[str, np.ndarray]:
        """
        Get summary of feature importance across all methods
        """
        return self.feature_scores_.copy()

    def get_selected_features_summary(self) -> Dict[str, np.ndarray]:
        """
        Get summary of selected features across all methods
        """
        return self.selected_features_.copy()


def optimize_feature_combination(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] = None,
    estimator_type: str = 'svm',
    max_features_range: Tuple[int, int] = (10, 50),
    step: int = 5,
    cv_folds: int = 3,
    scoring: str = 'roc_auc'
) -> Dict[str, Union[int, float, List]]:
    """
    Optimize number of features for best performance

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        estimator_type: Type of estimator
        max_features_range: Range of features to test
        step: Step size for feature range
        cv_folds: Number of CV folds
        scoring: Scoring metric

    Returns:
        Dict with optimal number of features and performance curve
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    selector = AdvancedFeatureSelector()

    # Preprocess features to handle NaN values
    X = selector._preprocess_features(X, fit_imputer=True)

    # Test different numbers of features
    n_features_list = list(range(max_features_range[0],
                                min(max_features_range[1], X.shape[1]) + 1,
                                step))
    scores_mean = []
    scores_std = []

    for n_features in n_features_list:
        # Use RFE to select features
        result = selector.recursive_feature_elimination(
            X, y, feature_names, estimator_type, n_features, cv_folds, use_cv=False
        )

        X_selected = result['X_selected']

        # Choose estimator for evaluation
        if estimator_type == 'rf':
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif estimator_type == 'svm':
            estimator = SVC(
                random_state=42,
                class_weight='balanced'
            )
        elif estimator_type == 'logreg':
            estimator = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown estimator: {estimator_type}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring=scoring)

        scores_mean.append(np.mean(cv_scores))
        scores_std.append(np.std(cv_scores))

    # Find optimal number of features
    best_idx = np.argmax(scores_mean)
    optimal_n_features = n_features_list[best_idx]
    optimal_score = scores_mean[best_idx]

    return {
        'optimal_n_features': optimal_n_features,
        'optimal_score': optimal_score,
        'n_features_list': n_features_list,
        'scores_mean': scores_mean,
        'scores_std': scores_std,
        'estimator_type': estimator_type
    }
