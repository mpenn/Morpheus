set(RAPIDS_CMAKE_VERSION "21.10" CACHE STRING "Version of rapids-cmake to use")

# Download and load the repo according to the rapids-cmake instructions if it does not exist
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
   message(STATUS "Downloading RAPIDS CMake Version: ${RAPIDS_CMAKE_VERSION}")
   file(
      DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_CMAKE_VERSION}/RAPIDS.cmake
      ${CMAKE_BINARY_DIR}/RAPIDS.cmake
   )
endif()

# Now load the file
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

# Load Rapids Cmake packages
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
