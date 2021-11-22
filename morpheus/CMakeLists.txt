list(APPEND CMAKE_MESSAGE_CONTEXT "_lib")

option(MORPHEUS_PYTHON_INPLACE_BUILD "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)

set(Python3_FIND_VIRTUALENV "FIRST")
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)


if(NOT EXISTS ${Python3_SITELIB}/skbuild)
  message(SEND_ERROR "Scikit-build is not installed. CMake may not be able to find Cython. Install scikit-build with `pip install scikit-build`")
else()
  list(APPEND CMAKE_MODULE_PATH "${Python3_SITELIB}/skbuild/resources/cmake")
endif()

# Now we can find dependencies
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Cython REQUIRED)
find_package(trtlab REQUIRED)

set(CYTHON_FLAGS "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True" CACHE STRING "The directives for Cython compilation.")

# # Command to copy the python files to the build folder
# add_custom_target(morpheus_python_rebuild ALL
#     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE_DIR:morpheus>/python)

# Save the root of the python for relative paths
set(MORPHEUS_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  message(STATUS "Performing inplace build for Python modules. Use `pip install -e .` from the python folder to install Neo.")
endif()

function(morpheus_get_python_module_name TARGET_NAME MODULE_NAME CPP_FILENAME)
  get_filename_component(py_module_name ${CPP_FILENAME} NAME_WLE)

  # Get the relative path from the py root
  file(RELATIVE_PATH py_rel_path ${MORPHEUS_PY_ROOT} ${CPP_FILENAME})
  get_filename_component(py_module_path ${py_rel_path} DIRECTORY)

  # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
  string(REPLACE "/" "." py_module_namespace ${py_module_path})

  set(${TARGET_NAME} "${py_module_namespace}.${py_module_name}" PARENT_SCOPE)
  set(${MODULE_NAME} "${py_module_name}" PARENT_SCOPE)
endfunction()

macro(morpheus_add_python_module source_file)

  set(prefix PYMOD)
  set(flags IS_PYBIND11 IS_CYTHON)
  set(singleValues MODULE_NAME OUTPUT_TARGET)
  set(multiValues LINK_TARGETS EXTRA_SOURCES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
                        "${flags}"
                        "${singleValues}"
                        "${multiValues}"
                        ${ARGN})

  # Get the full path to the cpp file
  file(REAL_PATH ${source_file} cpp_file)

  # Set the link libraries
  set(link_libs trtlab::pyneo)

  # Auto determine the module name
  morpheus_get_python_module_name(TARGET_NAME MODULE_NAME ${cpp_file})

  if (PYMOD_MODULE_NAME)
    # Overwrite the module name
    set(TARGET_NAME ${PYMOD_MODULE_NAME})
  endif()

  if (PYMOD_LINK_TARGETS)
    foreach(target IN LISTS PYMOD_LINK_TARGETS)
      list(APPEND link_libs ${target})
    endforeach()
  endif()

  # Actually create the target
  if (PYMOD_IS_PYBIND11)
    pybind11_add_module(${TARGET_NAME} MODULE ${cpp_file} ${PYMOD_EXTRA_SOURCES})
  elseif(PYMOD_IS_CYTHON)
    add_cython_target(${MODULE_NAME} "${cpp_file}" CXX PY3)
    add_library(${TARGET_NAME} SHARED ${${MODULE_NAME}} ${PYMOD_EXTRA_SOURCES})

    # # Need to set -fvisibility=hidden for cython according to https://pybind11.readthedocs.io/en/stable/faq.html
    # set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  else()
    message(FATAL_ERROR "Must specify either IS_PYBIND11 or IS_CYTHON")
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")
  target_link_libraries(${TARGET_NAME} PUBLIC ${link_libs})
  target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

  # Cython targets need the current dir for generated files
  if(PYMOD_IS_CYTHON)
    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  if(MORPHEUS_PYTHON_INPLACE_BUILD)
    # Get the source directory for the target
    get_filename_component(cpp_source_dir ${cpp_file} DIRECTORY)

    # If specified, copy files back to source for inplace build
    add_custom_command(TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${cpp_source_dir}
      COMMENT "Moving python target ${TARGET_NAME} to ${cpp_source_dir} for inplace build"
      # BYPRODUCTS "${cpp_source_dir}/$<TARGET_FILE_NAME:${TARGET_NAME}>" # Uncomment if genex can be used in BYPRODUCTS. See https://gitlab.kitware.com/cmake/cmake/-/issues/21364
    )
  endif()

  install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

  # add_dependencies(morpheus_python_rebuild ${TARGET_NAME})

  # Set the output target if desired
  if (PYMOD_OUTPUT_TARGET)
    set(${PYMOD_OUTPUT_TARGET} "${TARGET_NAME}" PARENT_SCOPE)
  endif()

endmacro()

function(morpheus_add_cython_libraries source_file)

  morpheus_add_python_module(${source_file} IS_CYTHON ${ARGN})

endfunction()

function(morpheus_add_pybind11_module source_file)

  morpheus_add_python_module(${source_file} IS_PYBIND11 ${ARGN})

endfunction()

add_subdirectory(_lib)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
