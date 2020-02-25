#!/bin/sh

echo "Using Sacred to log experiments\n"
python main.py --name cpc_audio \
    with \
    data_input_dir=./datasets/audio \
    num_epochs=300 \
    start_epoch=0 \
    fp16=False \
    fp16_opt_level="O2"