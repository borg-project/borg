#
# make cython easier
#

function(find_file_nocache v name)
    foreach(path ${ARGN})
        if(EXISTS "${path}/${name}")
            set(${v} "${path}/${name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${v} "${v}-NOTFOUND" PARENT_SCOPE)
endfunction()

function(add_cython_module target_name base_name)
    if(NOT DEFINED CYTHON_EXECUTABLE)
        message(SEND_ERROR "CYTHON_EXECUTABLE is not defined.")
    endif()

    # grab the corresponding declarations file, if it exists
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pxd)
        set(module_depends ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pxd)
    endif()

    # locate and add scanned dependencies
    include(${CMAKE_CURRENT_BINARY_DIR}/${base_name}.depends.cmake OPTIONAL)
    list(APPEND module_depends ${scanned})

    message(STATUS "module depends: ${module_depends}")

    # build up the -I arguments
    foreach(path ${CYTHON_INCLUDE_PATHS})
       list(APPEND CYTHON_INCLUDE_ARGUMENTS "-I${path}")
    endforeach()

    # create the build rule
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
        COMMAND
            ${CYTHON_EXECUTABLE}
            ${CYTHON_INCLUDE_ARGUMENTS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx
            -o
            ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.c
        COMMAND
            ${CMAKE_SOURCE_DIR}/src/scripts/scan_pyx_depends
            ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx
            ${CMAKE_CURRENT_BINARY_DIR}/${base_name}.depends.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CYTHON_INCLUDE_PATHS}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${base_name}.pyx
            ${module_depends}
        )
    add_library(${target_name} SHARED ${base_name}.c)
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${base_name} PREFIX "")
endfunction(add_cython_module)

function(cython_include_directories)
    set(CYTHON_INCLUDE_PATHS ${ARGN} ${CYTHON_INCLUDE_PATHS} PARENT_SCOPE)

    #foreach(path ${ARGN})
    #    list(APPEND CYTHON_INCLUDE_DIRECTORIES "-I${path}")
    #endforeach(path ${ARGN})
endfunction(cython_include_directories)

