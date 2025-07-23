#!/bin/bash

path="../../data/dataset"
dims=($(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
dims=("128" "300" "420" "960" "1024" "1536" "3072")
thread=40
# Loop over dimensions
for run in {1..2}; do
    echo "======== Pass $run ========"
    for DIM in "${dims[@]}"; do
        folder="$path/$DIM"
        # Find unique dataset names from *_base.fbin files
        mapfile -t datasets < <(find "$folder" -name '*_base.fbin' -exec basename {} \; | sed 's/_base\.fbin$//' | sort -u)

        for dataset in "${datasets[@]}"; do
            echo "Running $path $dataset $DIM"
            export OMP_NUM_THREADS=$thread
            ./bin/query "$path" "$dataset" "$DIM" "$thread"
        done
    done
done