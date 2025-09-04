#!/usr/bin/env python3
"""
Test script untuk memverifikasi bahwa feature selection dapat menangani NaN values
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add pipeline to path
sys.path.append('/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2')

from pipeline.feature_selection import AdvancedFeatureSelector, optimize_feature_combination

def test_nan_handling():
    """Test NaN handling in feature selection"""
    print("🔧 Testing NaN handling in feature selection...")

    # Create synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    # Introduce NaN values
    rng = np.random.RandomState(42)
    nan_indices = rng.choice(X.size, size=int(0.1 * X.size), replace=False)
    X_flat = X.flatten()
    X_flat[nan_indices] = np.nan
    X_with_nan = X_flat.reshape(X.shape)

    print(f"Original shape: {X.shape}")
    print(f"NaN values introduced: {np.sum(np.isnan(X_with_nan))}")
    print(f"NaN percentage: {np.sum(np.isnan(X_with_nan))/X_with_nan.size*100:.1f}%")

    # Test each feature selection method
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    selector = AdvancedFeatureSelector(random_state=42)

    print("\n📊 Testing feature selection methods:")

    try:
        # Test univariate selection
        print("  ✅ Testing univariate selection...")
        result_uni = selector.univariate_selection(
            X_with_nan, y, feature_names, method='f_classif', k=10
        )
        print(f"     Selected {len(result_uni['selected_features'])} features")

        # Test RFE
        print("  ✅ Testing RFE...")
        result_rfe = selector.recursive_feature_elimination(
            X_with_nan, y, feature_names, estimator_type='rf', n_features=10
        )
        print(f"     Selected {len(result_rfe['selected_features'])} features")

        # Test model-based
        print("  ✅ Testing model-based selection...")
        result_model = selector.model_based_selection(
            X_with_nan, y, feature_names, estimator_type='rf'
        )
        print(f"     Selected {len(result_model['selected_features'])} features")

        # Test ensemble
        print("  ✅ Testing ensemble selection...")
        result_ensemble = selector.ensemble_feature_selection(
            X_with_nan, y, feature_names, top_k=10
        )
        print(f"     Selected {len(result_ensemble['selected_features'])} features")

        # Test optimization
        print("  ✅ Testing feature optimization...")
        result_opt = optimize_feature_combination(
            X_with_nan, y, feature_names,
            estimator_type='rf',
            max_features_range=(5, 15),
            step=2
        )
        print(f"     Optimal features: {result_opt['optimal_n_features']}")
        print(f"     Optimal score: {result_opt['optimal_score']:.4f}")

        print("\n🎉 All feature selection methods handled NaN values successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error in feature selection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_infinite_values():
    """Test handling of infinite values"""
    print("\n🔧 Testing infinite values handling...")

    # Create dataset with infinite values
    X, y = make_classification(n_samples=50, n_features=10, random_state=42)

    # Introduce infinite values
    X[5, 2] = np.inf
    X[10, 7] = -np.inf
    X[15, 3] = np.nan

    print(f"Infinite values: {np.sum(np.isinf(X))}")
    print(f"NaN values: {np.sum(np.isnan(X))}")

    try:
        selector = AdvancedFeatureSelector(random_state=42)
        result = selector.ensemble_feature_selection(X, y, top_k=5)
        print(f"✅ Successfully selected {len(result['selected_features'])} features")
        print("🎉 Infinite values handled correctly!")
        return True
    except Exception as e:
        print(f"❌ Error handling infinite values: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting feature selection NaN handling tests...\n")

    # Run tests
    test1_passed = test_nan_handling()
    test2_passed = test_infinite_values()

    print(f"\n📋 Test Results:")
    print(f"  NaN handling: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Infinite values: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Feature selection is ready for NaN handling.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
