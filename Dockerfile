# Use the official Ubuntu base image
FROM ubuntu:latest

# Install required packages
RUN apt-get update && \
    apt-get install -y git cmake ninja-build clang

# Set the working directory
WORKDIR /llvm-project

# Clone the LLVM project repository
RUN git clone https://github.com/tridhapuku/DSP_MLIR.git .

# Checkout latest Branch
RUN git checkout latestMain

# Create the build directory
RUN mkdir build

# Change to the build directory
WORKDIR /llvm-project/build

# Download and install cmake
RUN apt-get install -y cmake

# Configure and build LLVM projects
RUN cmake -S llvm -B build -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=OFF

# Configure and build MLIR project
RUN cmake ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON

# Build and run MLIR tests
RUN cmake --build . --target check-mlir
