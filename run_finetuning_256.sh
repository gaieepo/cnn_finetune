#!/bin/bash
TIME=$(date +%Y%m%d_%H%M)

echo "Fold 01"
(time CUDA_VISIBLE_DEVICES=1 python3 densenet121_cribriform.py -f 1) &> logs/log_fold_1_$TIME.log

echo "Fold 02"
(time CUDA_VISIBLE_DEVICES=1 python3 densenet121_cribriform.py -f 2) &> logs/log_fold_2_$TIME.log

echo "Fold 03"
(time CUDA_VISIBLE_DEVICES=1 python3 densenet121_cribriform.py -f 3) &> logs/log_fold_3_$TIME.log

echo "Fold 04"
(time CUDA_VISIBLE_DEVICES=1 python3 densenet121_cribriform.py -f 4) &> logs/log_fold_4_$TIME.log

# CUDA_VISIBLE_DEVICES=1,2
