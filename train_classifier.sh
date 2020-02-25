#!/bin/sh

python -m testing.logistic_regression_phones \
    with \
    model_path=./logs/1 \
    model_num=300 \
    fp16=False