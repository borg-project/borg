#
# locate numpy
#

if(NOT NUMPY_INCLUDE_DIRS)
    find_package(PythonInterp REQUIRED)

    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/echo_numpy_path.py"
        OUTPUT_VARIABLE NUMPY_FOUND_PATH
        ERROR_VARIABLE NUMPY_FOUND_PATH
        RESULT_VARIABLE ECHO_NUMPY_PATH_FAILED
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    set(NUMPY_INCLUDE_DIRS ${NUMPY_FOUND_PATH}/core/include)
endif(NOT NUMPY_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Numpy DEFAULT_MSG NUMPY_INCLUDE_DIRS)

