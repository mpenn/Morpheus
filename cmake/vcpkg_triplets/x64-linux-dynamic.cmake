set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Set the RPATH to allow finding other libraries in the same folder
set(VCPKG_LINKER_FLAGS "-Wl,-rpath,'$ORIGIN':'$ORIGIN/../lib'")
