#!/bin/bash

# ✅ FIXED: Command yang Benar untuk Pipeline Testing

echo "🚀 Testing Complete EEG Pipeline with Feature Selection..."
echo "Error 'unrecognized argument' sudah diperbaiki!"
echo ""

# QUICK TEST - Validasi Fix
echo "1️⃣ Quick Test (untuk validasi):"
echo "python -m pipeline.experiments \\"
echo "  --root /Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2 \\"
echo "  --max_channels_list \"3\" \\"
echo "  --balance_list oversample \\"
echo "  --add_time --add_stft \\"
echo "  --feature_selection --fs_method ensemble --fs_k 10 \\"
echo "  --limit_train 10 --limit_eval 10 \\"
echo "  --export_dir outputs_quick_test \\"
echo "  --repeats 1"
echo ""

# MEDIUM TEST
echo "2️⃣ Medium Test (hasil yang bagus):"
echo "python -m pipeline.experiments \\"
echo "  --root /Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2 \\"
echo "  --max_channels_list \"3,5\" \\"
echo "  --balance_list oversample \\"
echo "  --add_time --add_stft \\"
echo "  --feature_selection --fs_method ensemble --fs_k 15 \\"
echo "  --limit_train 50 --limit_eval 50 \\"
echo "  --export_dir outputs_medium_test \\"
echo "  --repeats 2 \\"
echo "  --stats --feature_significance"
echo ""

# FULL OPTIMIZATION
echo "3️⃣ Full Test dengan Optimization (hasil terbaik):"
echo "python -m pipeline.experiments \\"
echo "  --root /Users/hilmania/Documents/Thesis/dataset/EEG_NEW_16CHS_2 \\"
echo "  --max_channels_list \"3,5,8\" \\"
echo "  --balance_list oversample \\"
echo "  --add_time --add_stft \\"
echo "  --feature_selection --fs_method ensemble \\"
echo "  --optimize_k_features --fs_k_range \"10,30\" \\"
echo "  --export_dir outputs_optimized \\"
echo "  --repeats 3 \\"
echo "  --stats --feature_significance"
echo ""

echo "📌 PENTING: Gunakan tanda kutip (\") untuk argumen yang mengandung koma!"
echo "   ❌ Salah: --max_channels_list 3 5 8"
echo "   ✅ Benar: --max_channels_list \"3,5,8\""
echo ""

echo "🎯 Mulai dengan Quick Test untuk memastikan semua berjalan normal."
