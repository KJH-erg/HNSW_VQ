
C=4096
thread_num=40
path="../../../data/dataset"
dims=($(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
dims=("128" "300" "420" "960" "1024" "1536" "3072")
for DIM in "${dims[@]}"; do
    D=$DIM
    B=$(( (D + 63) / 64 * 64 ))
    folder="$path/$DIM"
    mapfile -t datasets < <(find "$folder" -name '*_base.fbin' -exec basename {} \; | sed 's/_base\.fbin$//' | sort -u)
    mkdir -p ../indices/$D
    for dataset in "${datasets[@]}"; do

        g++ -o ../bin/index_${D}_${dataset} ../src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
        "../bin/index_${D}_${dataset}" "$path" "$dataset" "$thread_num"
    done
done
