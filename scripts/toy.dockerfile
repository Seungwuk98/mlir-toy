FROM ubuntu:24.04 

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y
RUN apt install -y software-properties-common wget 

RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
RUN add-apt-repository "deb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-18 main" -y
RUN apt update 

RUN apt install -y cmake ninja-build clang clang-tools lld llvm-18-dev libmlir-18-dev mlir-18-tools git zlib1g-dev libzstd-dev

COPY . /root/mlir-toy

WORKDIR /root/mlir-toy

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
        -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \ 
        -DLLVM_OPTIMIZED_TABLEGEN=TRUE 

RUN cmake --build build
RUN cmake --install build 

WORKDIR /root

CMD ["/bin/bash"]
