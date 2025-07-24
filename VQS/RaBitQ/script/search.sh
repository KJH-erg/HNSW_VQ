
C=4096


thread_num=40
path="../../../data/dataset"
dims=($(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort))
dims=("128" "300" "420" "960" "1024" "1536" "3072")
# dims=("300")
for DIM in "${dims[@]}"; do
    D=$DIM
    B=$(( (D + 63) / 64 * 64 ))
    folder="$path/$DIM"
    mapfile -t datasets < <(find "$folder" -name '*_base.fbin' -exec basename {} \; | sed 's/_base\.fbin$//' | sort -u)
    mkdir -p ../indices/$D
    for dataset in "${datasets[@]}"; do

        g++ -march=core-avx2 -fopenmp -o ../bin/search_${D}_${dataset} ../src/search.cpp -I ./src/ -D BB=${B} -D DIMENSION=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
        echo "Running $BIN_PATH 3 times..."
        for i in {1..3}; do
            echo "Run $i for $dataset at D=$D"
            "../bin/search_${D}_${dataset}" "$path" "$dataset" "$thread_num"
        done
    done
done

# g++ -march=core-avx2 -Ofast -o ../bin/search_${data} ../src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN

        # g++ -march=core-avx2 -Ofast -fopenmp -o ../bin/search ../src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
# g++ -march=core-avx2 -g -o ../bin/search ../src/create_index.cpp -I ./src/ \
# -D BB=${B} -D DIMENSION=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
# result_path=./results
# mkdir ${result_path}

# res="${result_path}/${data}/"

# mkdir "$result_path/${data}/"

# ./bin/search -d ${data} -r ${res} -k ${k} -s "$source/$data/"
