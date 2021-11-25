#!/bin/sh

python -m testing.logistic_regression_speaker \
    with \
    data_input_dir=/groups/1/gcc50521/furukawa/musicnet_npy_10sec \
    model_path=/groups/1/gcc50521/furukawa/cpc_logs/26 \
    model_num=450 \
    fp16=False

# python -m testing.logistic_regression_phones \
#     with \
#     model_path=./logs/1 \
#     model_num=300 \
#     fp16=False
