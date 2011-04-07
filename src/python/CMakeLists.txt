#
# configure python dependencies
#

find_package(Numpy REQUIRED)

execute_process(
    COMMAND
        "python"
        "${PROJECT_SOURCE_DIR}/src/scripts/echo_python_inc_path.py"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

message(STATUS "Python include path: ${PYTHON_INCLUDE_DIRS}")

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS} # should be last
    )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")

#
# configure cython
#

find_package(Cython REQUIRED)

include(UseCython)

#
# source
#

install(
    FILES
        cargo_site_defaults.py
    DESTINATION lib
    )

#
# descend
#

add_subdirectory(borg)

