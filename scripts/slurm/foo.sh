#!/bin/bash

# module purge
# module load gcc/9.2.0
# module load python-miniconda3/4.12.0
# source activate hiec
# python foo.py

for i in $(seq 0 9);
do
    echo $((($i % 2) * -1));
done

