#!/bin/sh

python -m testing.logistic_regression_speaker \
    with \
    model_path=./logs/cpc_audio_baseline \
    model_num=299 \
    fp16=False

# python -m testing.logistic_regression_phones \
#     with \
#     model_path=./logs/1 \
#     model_num=300 \
#     fp16=False
