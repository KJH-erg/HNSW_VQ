### We used DiskANN's function to calculate ground truth values of K nearest neighbors
### Install required libraries
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev
### Clone DiskANN git
git clone https://github.com/microsoft/DiskANN.git
### Install DiskANN
cd DiskANN
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 