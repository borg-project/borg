#
# locate Python headers
#

if(NOT PYTHON_INCLUDE_DIRS)
    execute_process(
        COMMAND
            "python"
            "${PROJECT_SOURCE_DIR}/src/scripts/echo_python_inc_path.py"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
        ERROR_VARIABLE PYTHON_INCLUDE_DIRS
        RESULT_VARIABLE ECHO_PYTHON_INC_PATH_FAILED
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
endif(NOT PYTHON_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(PythonHeaders DEFAULT_MSG PYTHON_INCLUDE_DIRS)

