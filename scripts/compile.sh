#!/bin/bash

set -x

# Optionally can set INSTALL_PREFIX to build and install to a specific directory. Also causes cmake install to run
BUILD_DIR=${BUILD_DIR:-"build"}

echo "Runing CMake configure..."
cmake -B ${BUILD_DIR} -GNinja \
   -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
   -DMORPHEUS_USE_CLANG_TIDY=OFF \
   -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
   -DMORPHEUS_USE_CCACHE=ON \
   ${INSTALL_PREFIX:+"-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"} .

echo "Running CMake build..."
cmake --build ${BUILD_DIR} -j ${INSTALL_PREFIX:+"--target install"}
