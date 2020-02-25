#!/bin/sh
conda env create -f environment.yml || conda env update -f environment.yml || exit

# for faster tsne
# conda install tsnecuda cuda101 -c cannylab
