#
# make cython easier
#

function(add_cython_module target_name base_name)
    add_custom_command(
       OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
       COMMAND cython ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx -o ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx
       )
    add_library(${target_name} SHARED ${base_name}.c)
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${base_name} PREFIX "")
endfunction(add_cython_module)

#
# configure python dependencies
#

find_package(Numpy REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(GSL REQUIRED)

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    )

#
# descend
#

add_subdirectory(borg)

