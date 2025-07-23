#!/bin/bash

path="../../data/dataset"
dims=($(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
thread=40
dims=("300" "420")
Bs=(3 4 5 7 8 9)

# Loop over dimensions
for run in {1..2}; do
    echo "======== Pass $run ========"

    for DIM in "${dims[@]}"; do
        for B in "${Bs[@]}"; do
            folder="$path/$DIM"
            mapfile -t datasets < <(find "$folder" -name '*_base.fbin' -exec basename {} \; | sed 's/_base\.fbin$//' | sort -u)

            for dataset in "${datasets[@]}"; do
                echo "Running $path $dataset $DIM (B=$B) - Pass $run"
                ./bin/create_index "$path" "$dataset" "$DIM" "$thread" "$B" 4096
            done
        done
    done
done