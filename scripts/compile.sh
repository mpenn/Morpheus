#!/bin/bash

echo "Runing CMake configure..."
cmake -B build -GNinja \
   -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
   -DMORPHEUS_USE_CLANG_TIDY=OFF \
   -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
   -DMORPHEUS_USE_CCACHE=OFF .

echo "Running CMake build..."
cmake --build build
