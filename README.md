# Toy MLIR

This project implements a simple programming language `Toy` which is using `MLIR` as its intermediate representation. The details of this project is from [MLIR tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) and specification is not changed. Full `Toy` implementation is in [llvm-project](https://github.com/llvm/llvm-project) and this project refers to that implementation. The purpose of this project is to understand how to use MLIR to implement a simple programming language. 


## Build 

This is **out-of-tree** project based on `llvm` and `mlir` project. So, you need to build `llvm` and `mlir` or download pre-built binaries. 

### Download pre-built binaries

If you are using Debian-based Linux, you can download pre-built binaries by running the following instructions. 

First, Add repository for `llvm` and `mlir` project. Refer to [here](https://apt.llvm.org/) for detail information. 

Second, install `llvm` and `mlir` project. We also need `cmake`, `ninja`, `clang`, and `lld`(optional) to build this project.  

```bash
$ apt install cmake ninja-build clang lld llvm-18-dev libmlir-18-dev mlir-18-tools
```

### Build from source

If you want to build `llvm` and `mlir` from source, you can refer to [here](https://llvm.org/docs/CMake.html). Following is the example of building `llvm` and `mlir` from source. 

```bash
$ git clone --depth 1 --branch llvmorg-18.0.0 https://github.com/llvm/llvm-project
$ cd llvm-project
$ cmake -S llvm -B llvm-build -G Ninja -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" -DCMAKE_BUILD_TYPE=Release
$ cmake --build llvm-build
$ cmake --install llvm-build
```


### Build Toy

After installing `llvm` and `mlir`, you can build `Toy` project. 

```bash
$ git clone https://github.com/Seungwuk98/mlir-toy.git
$ cd mlir-toy
$ git submodule update --recursive --init
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=${build type} \
        -DLLVM_DIR=${llvm build directory}/lib/cmake/llvm \
        -DMLIR_DIR=${mlir build directory}/lib/cmake/mlir \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \ # if you want to use lld
        -DLLVM_OPTIMIZED_TABLEGEN=ON # if you want to use optimized tablegen
$ cmake --build build
$ cmake --install build # if you want to install toy
```

If you want to use alias of `toyc` and `toy-opt`, export `PATH` environment variable or create alias. 

```bash
$ export PATH=$PATH:/path/to/mlir-toy/build/tools
```

```bash
$ alias toyc=/path/to/mlir-toy/build/tools/toyc
$ alias toy-opt=/path/to/mlir-toy/build/tools/toy-opt
```

### Docker 

`Dockerfile` is already provided in this project. You can build docker image by running the following commands. 

```bash 
mlir-toy$ docker build -t mlir-toy -f scripts/toy.dockerfile .
mlir-toy$ docker run -it --rm mlir-toy
```


## Run 

After building `Toy` project, you can run `Toy` program. 

```toy 
# example.toy
def transpose_multiply(a, b) {
    return transpose(a) * transpose(b);
}
def main() {
    var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
    var b<2, 3> = [[6, 5, 4], [3, 2, 1]]; 
    var c = transpose_multiply(a, b);
    print(c);
}
```

You can compile `Toy` program to `MLIR` by running the following command.

```bash 
$ toyc -action=toy example.toy -o example.mlir 
```

`-action=toy` means that output MLIR is in `Toy Dialect`.

```mlir
// example.mlir

module {
  toy.func private @transpose_multiply(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.constant dense<[[6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<2x3xf64>
    %3 = toy.reshape(%2 : tensor<2x3xf64>) to tensor<2x3xf64>
    %4 = toy.generic_call @transpose_multiply(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %4 : tensor<*xf64>
    toy.return
  }
}
```

You can also optimize (or lowered) program by adjusting `-action` option. 

```bash
$ toyc -action=affine example.toy -o example.mlir  # Lower to Affine Dialect
$ toyc -action=llvm example.toy -o example.mlir    # Lower to LLVM Dialect
$ toyc -action=llvm-ir example.toy -o example.ll   # Lower to LLVM IR
```

You can run `Toy` program by using JIT compilation.

```bash
$ toyc -action=jit example.toy
6.000000 12.000000
10.000000 10.000000
12.000000 6.000000
```

Or compile `.ll` file to executable file.  


```bash
$ toyc -action=llvm-ir example.toy -o example.ll 
$ clang example.ll -o example 
$ ./example
6.000000 12.000000
10.000000 10.000000
12.000000 6.000000
```

## Etc 

If you want to see the intermediate representation of `Toy` program, you can use `toy-opt` tool. 

```bash
$ toy-opt --inline example.mlir -o example-inline.mlir
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %3 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %6 = toy.mul %4, %5 : tensor<*xf64>
    toy.print %6 : tensor<*xf64>
    toy.return
  }
}
```

All of MLIR default passes are available in `toy-opt` tool and `Toy` specific passes are also available.

`--toy-shape-inference` : Infer shape of tensors.
`--toy-print-lowering` : Lower `toy.print` operation to `SCF` and `LLVM` dialect. Ensure that the program must be lowered to affine dialect. 
`--toy-to-affine` : Lower `Toy` dialect to `Affine` dialect.
`--toy-affine-to-llvm` : Lower `Affine` dialect to `LLVM` dialect

