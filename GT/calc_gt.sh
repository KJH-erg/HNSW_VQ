#!/bin/bash

dataset_path="../data/dataset"
diskann_root="DiskANN/build/apps/utils"

dims=($(find "$dataset_path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
dims=("128" "300" "420" "960" "1024" "1536" "3072")
for dim in "${dims[@]}"; do
    dim_path="${dataset_path}/${dim}"

    # Collect unique dataset names (before the first underscore)
    datas=()
    for file in "$dim_path"/*_base.*; do
        [ -e "$file" ] || continue  # skip if no matches
        filename=$(basename "$file")
        name="${filename%%_*}"  # take prefix before first underscore
        datas+=("$name")
    done

    unique_datas=($(printf "%s\n" "${datas[@]}" | sort -u))
    for name in "${unique_datas[@]}"; do
        if [ "$name" == "tempfolder" ]; then
            continue
        fi

        echo "Compute GT for dimension $dim with data $name"

        "${diskann_root}/compute_groundtruth" \
            --data_type float --dist_fn l2 \
            --base_file "${dataset_path}/${dim}/${name}_base.fbin" \
            --query_file "${dataset_path}/${dim}/${name}_query.fbin" \
            --gt_file "${dataset_path}/${dim}/${name}_gt.fbin" \
            --K 1000
    done
done