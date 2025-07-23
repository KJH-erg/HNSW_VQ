# Introduction
This repository contains experiments evaluating the impact of different quantization methods on HNSW
# Directory Information
<pre>
<code>
HNSW_VQ/
├── indices
├── logs
|   └── build_log.csv
|   └── search_log.csv
├── data
|   └── download.sh
|   └── dataset/
|          └── dim/
|               └── fbin files
|
├── GT
|   └──  calc_gt.sh
|
├── FLOAT
|   ├── src/
│   |    └── create_index.cpp
|   |    └── query.cpp
|   └── build.sh
|   └── query.sh
|
├── VQS
|   ├── PQ/
|   |    ├── src/
│   |    |   └── create_index.cpp
|   |    |   └── query.cpp
|   |    └── build.sh
|   |    └── query.sh
|   |
|   ├── SQ/
|   |    ├── src/
│   |    |   └── create_index.cpp
|   |    |   └── query.cpp
|   |    └── build.sh
|   |    └── query.sh
|   |
|   ├── EXrabitq/
|   |    ├── data/
|   |    ├── logs/
|   |    ├── src/
│   |    |   └── create_index.cpp
|   |    |   └── query.cpp
|   |    ├── python/
|   |    |   └── utils
|   |    |   └── ivf.py
|   |    └── ivf.sh
|   |    └── build.sh
└── README.md
</code>
</pre>
# Prepare Test Env
## Prepare python env
<pre>
<code>
conda create -n test_env python=3.10 numpy -y
conda activate test_env
pip install faiss-cpu scikit_learn datasets
</code>
</pre>
## Install Faiss
The Faiss library is employed to compute codes for both Product Quantization (PQ) and Scalar Quantization (SQ).
<pre>
<code>
cd third 
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_USE_STATS=ON -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF
cmake --build build -j
</code>
</pre>
# Dataset Preparation
In the `data/` directory, the `download.sh` script automatically downloads several public datasets and converts them into `.fbin` format for ANN experiments. The supported datasets include:

- **SIFT (128)**
- **Word2Vec (300)**
- **MSong (420)**
- **GIST (960)**
- **YouTube (1024)**

## How to Run
<pre>
<code>
cd data
./download.sh
</code>
</pre>

## Dataset Directory Structure

The datasets are stored under the `data/dataset/` directory, with each dataset placed in a folder named after its vector dimensionality (e.g., `128`, `300`, `960`, etc.).

Each folder typically contains:

- `<name>_base.fbin` — base vectors used for indexing
- `<name>_query.fbin` — query vectors used for search

# Prepare ground Truth values
DiskANN’s utility is used to compute the top-1000 ground truth nearest neighbors for each query vector.
## How to Install DiskANN
<pre>
<code>
cd GT
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
sudo apt install libmkl-full-dev
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
</code>
</pre>
### How to Run
<pre>
<code>
cd GT
./calc_gt.sh
</code>
</pre>

# Run HNSW with original Float data and Save index
This folder contains the files necessary to build an HNSW index using float data and to perform baseline evaluations.
## Configuration
- `M = 32`
- `efConstruction = 200`
- `threads = 60`

Index build statistics such as `dimension`, `dataset`, `index_build_time`, and `peak_memory_mb` are recorded in `logs/build_log.csv`
## How to compile files
<pre>
<code>
cd FLOAT
mkdir build
cmake ..
make -j
</code>
</pre>
## How to build Index
<pre>
<code>
cd FLOAT
./build.sh
</code>
</pre>

The resulting index is saved in the `indices/float` folder, with each subfolder named after the dimensionality of the dataset (e.g., `128/`, `300/`, `960/`, etc.).

## How to run baseline evaluation
<pre>
<code>
cd FLOAT
./build.sh
</code>
</pre>

## Configuration
- `ef_search= 10~200` with step 10
- `threads = 40`
Search statistics including `method`, `dataset`, `dim`, `peak_memory_mb`, `vec_size`, `ef_search`, `qps`, `n_cmp`, `recall@1`, `recall@10`, `recall@100`, and `recall@1000` — are recorded in `logs/search_log.csv`.

# Run HNSW with PQ data and Save index
## How to compile files
<pre>
<code>
cd VQS/PQ
mkdir build
cmake ..
make -j
</code>
</pre>
## How to re-construct HNSW index with PQ data
<pre>
<code>
cd VQS/PQ
./build.sh
</code>
</pre>
The resulting index is saved in the `indices/PQ` folder, with each subfolder named after the dimensionality of the dataset (e.g., `128/`, `300/`, `960/`, etc.).  
Within each subfolder, the index files are named according to the number of subvectors used for Product Quantization (e.g., `16_sift_hnsw.index`, `64_sift_hnsw.index`, etc.).
## How to run PQ evaluation
<pre>
<code>
cd VQS/PQ
./query.sh
</code>
</pre>

## Configuration
- `ef_search= 10~200` with step 10
- `threads = 40`

Search statistics including `method`, `dataset`, `dim`, `peak_memory_mb`, `vec_size`, `ef_search`, `qps`, `n_cmp`, `recall@1`, `recall@10`, `recall@100`, and `recall@1000` — are recorded in `logs/search_log.csv`.


# Run HNSW with SQ data and Save index
## How to compile files
<pre>
<code>
cd VQS/SQ
mkdir build
cmake ..
make -j
</code>
</pre>
## How to re-construct HNSW index with SQ data
<pre>
<code>
cd VQS/SQ
./build.sh
</code>
</pre>
The resulting index is saved in the `indices/SQ` folder, with each subfolder named after the dimensionality of the dataset (e.g., `128/`, `300/`, `960/`, etc.).  
Within each bits per dimension, the index files are named according to the number of subvectors used for Product Quantization (e.g., `4_sift_hnsw.index`, `6_sift_hnsw.index`, etc.).
## How to run SQ
<pre>
<code>
cd VQS/SQ
./query.sh
</code>
</pre>

## Configuration
- `ef_search= 10~200` with step 10
- `threads = 40`

Search statistics including `method`, `dataset`, `dim`, `peak_memory_mb`, `vec_size`, `ef_search`, `qps`, `n_cmp`, `recall@1`, `recall@10`, `recall@100`, and `recall@1000` — are recorded in `logs/search_log.csv`.


# Run HNSW with RabitQ data 
## How to compile files
<pre>
<code>
cd VQS/EXrabitq
mkdir build
cmake ..
make -j
</code>
</pre>
## Preliminary Clustering for RabitQ 
It runs the Faiss clustering algorithm using Python code located in the `python` folder as a preliminary step for RabitQ.
<pre>
<code>
cd VQS/EXrabitq
./ivf.sh
</code>
</pre>
Generated clustering information is saved in `data` folder with each subfolder named after the dimensionality of the dataset (e.g., `128/`, `300/`, `960/`, etc.).  
## Generate RabitQ code and run HNSW with constructed code
<pre>
<code>
cd VQS/EXrabitq
./build.sh
</code>
</pre>
## Configuration
- `ef_search= 10~200` with step 10
- `threads = 40`

Search statistics including `method`, `dataset`, `dim`, `peak_memory_mb`, `vec_size`, `ef_search`, `qps`, `n_cmp`, `recall@1`, `recall@10`, `recall@100`, and `recall@1000` — are recorded in `logs/search_log.csv`.