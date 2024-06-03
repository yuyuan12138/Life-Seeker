#!/bin/bash

# for i in {1..20}; do
#     python -u train.py
# done

for dir in ./data/*; do
    for i in {6..20}; do
        python -u train.py -data ${dir##*/} -loop $i
    done
done
