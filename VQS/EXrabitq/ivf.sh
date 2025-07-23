
path="../../data/dataset"
dims=($(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
dims=("420")
mkdir -p logs
Bs=(3 4 5 7 8 9)
# Loop over dimensions
for DIM in "${dims[@]}"; do
    folder="$path/$DIM"

    # Find unique dataset names from *_base.fbin files
    mapfile -t datasets < <(find "$folder" -name '*_base.fbin' -exec basename {} \; | sed 's/_base\.fbin$//' | sort -u)
    
    for dataset in "${datasets[@]}"; do
        echo "Running $path $dataset $DIM"
        python3 python/ivf.py $path $DIM $dataset

    done
done