#
# locate cython
#

if(NOT CYTHON_EXECUTABLE)
    find_program(CYTHON_EXECUTABLE cython)
endif(NOT CYTHON_EXECUTABLE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Cython DEFAULT_MSG CYTHON_EXECUTABLE)

