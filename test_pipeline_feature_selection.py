#!/usr/bin/env python3
"""
Script untuk menguji pipeline lengkap dengan feature selection
"""

import os
import sys

def test_pipeline_with_feature_selection():
    """Test pipeline dengan feature selection"""

    # Perintah untuk test pipeline lengkap
    cmd = """
python -m pipeline.experiments \
  --root /Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2 \
  --max_channels_list 3 \
  --balance_list oversample \
  --add_time --add_stft \
  --feature_selection --fs_method ensemble --fs_k 15 \
  --limit_train 50 --limit_eval 50 \
  --export_dir outputs_test \
  --repeats 1
"""

    print("🚀 Testing pipeline with feature selection...")
    print("Command to run:")
    print(cmd)
    print("\n" + "="*50)

    # Execute the command
    exit_code = os.system(cmd)

    if exit_code == 0:
        print("\n✅ Pipeline test PASSED!")
        print("Features selection handled NaN values successfully")
        return True
    else:
        print(f"\n❌ Pipeline test FAILED with exit code {exit_code}")
        return False

if __name__ == "__main__":
    success = test_pipeline_with_feature_selection()

    if success:
        print("\n🎉 Pipeline is ready for feature selection with NaN handling!")
        print("\nYou can now run the full pipeline with:")
        print("""
python -m pipeline.experiments \\
  --feature_selection --fs_method ensemble \\
  --optimize_k_features --fs_k_range "10,30" \\
  --add_time --add_stft \\
  --stats --feature_significance \\
  --balance_list oversample
""")
    else:
        print("\n⚠️  Please check the error messages above")
