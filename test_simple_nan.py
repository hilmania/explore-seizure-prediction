#!/usr/bin/env python3
"""
Simple test untuk feature selection dengan NaN values
"""

import numpy as np
import sys
import os

# Add pipeline to path
sys.path.append('/Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2')

try:
    from pipeline.feature_selection import AdvancedFeatureSelector
    print("✅ Successfully imported AdvancedFeatureSelector")

    # Create simple test data with NaN
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
        [np.nan, 11.0, 12.0]
    ])
    y = np.array([0, 1, 0, 1])

    print(f"Test data shape: {X.shape}")
    print(f"NaN count: {np.sum(np.isnan(X))}")

    # Test selector
    selector = AdvancedFeatureSelector(random_state=42)

    # Test preprocessing function
    X_clean = selector._preprocess_features(X, fit_imputer=True)
    print(f"✅ Preprocessing successful")
    print(f"NaN count after preprocessing: {np.sum(np.isnan(X_clean))}")
    print(f"Cleaned data:\n{X_clean}")

    # Test univariate selection
    result = selector.univariate_selection(X, y, k=2)
    print(f"✅ Univariate selection successful")
    print(f"Selected features: {result['selected_indices']}")

    print("\n🎉 Basic NaN handling test PASSED!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
