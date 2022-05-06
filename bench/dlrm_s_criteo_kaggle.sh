#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
echo $dlrm_extra_option

dlrm_pt_bin="python dlrm_pytorch.py"

echo "run pytorch ..."

#Data param
data=dataset
data_type=kaggle
input_path="/disk/scratch/s2189665/dlrm/input"
output_path="/disk/scratch/s2189665/dlrm/output_original"
mkdir -p $input_path
mkdir -p $output_path
raw_data_file=$input_path/train.txt
processed_data_file=$input_path/kaggleAdDisplayChallenge_processed.npz
log_file=$output_path/run_kaggle_pt.log

#Train param
loss_func=bce
round_targets=True
lr=0.1
m_batch_size=128
test_m_batch_size=16384
test_n_workers=0
print_freq=1024
test_freq=10240
save_model_file=$output_path/"best_kaggle_model.pt"

#Model param
sparse_size=16
bot_mlp="13-512-256-64-16"
top_mlp="512-256-1"


_args="  --arch-sparse-feature-size="${sparse_size}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --data-generation="${data}\
" --data-set="${data_type}\
" --raw-data-file="${raw_data_file}\
" --processed-data-file="${processed_data_file}\
" --loss-function="${loss_func}\
" --round-targets="${round_targets}\
" --learning-rate="${lr}\
" --mini-batch-size="${m_batch_size}\
" --test-freq="${test_freq}\
" --test-mini-batch-size="${test_m_batch_size}\
" --test-num-workers="${test_n_workers}\
" --print-freq="${print_freq}\
" --print-time"\
" --save-model="${save_model_file}\
" --use-gpu"\

export CUDA_VISIBLE_DEVICES=1
cmd="$dlrm_pt_bin $_args $dlrm_extra_option 2>&1 | tee $log_file"

# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# cmdo="$dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=$raw_data_file --processed-data-file=$processed_data_file --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee $log_file"
echo $cmd
# echo $cmdo
eval $cmd

# echo "run caffe2 ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# $dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

echo "done"
