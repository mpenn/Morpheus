#!/bin/bash

BUILD_DIR=${BUILD_DIR:-"build"}

echo "Runing CMake configure..."
cmake -B ${BUILD_DIR} -GNinja \
   -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
   -DMORPHEUS_USE_CLANG_TIDY=OFF \
   -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
   -DMORPHEUS_USE_CCACHE=ON .

echo "Running CMake build..."
cmake --build ${BUILD_DIR} -j
