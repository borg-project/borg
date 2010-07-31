#
# make cython easier
#

function(add_cython_module target_name base_name)
    if(NOT DEFINED CYTHON_EXECUTABLE)
        message(SEND_ERROR "Cython executable CYTHON_EXECUTABLE is not defined.")
    endif(NOT DEFINED CYTHON_EXECUTABLE)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pxd)
        set(${more_depends} ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pxd)
    endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pxd)

    add_custom_command(
       OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
       COMMAND ${CYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx -o ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx ${more_depends}
       )
    add_library(${target_name} SHARED ${base_name}.c)
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${base_name} PREFIX "")
endfunction(add_cython_module)

