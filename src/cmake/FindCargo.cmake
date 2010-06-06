#
# locate cargo
#

if(NOT CARGO_LIBRARIES OR NOT CARGO_INCLUDE_DIRS)
    find_path(CARGO_INCLUDE_DIR utexas/python/numpy.h)
    find_library(CARGO_LIBRARY_PYTHON ut_python)

    set(CARGO_INCLUDE_DIRS ${CARGO_INCLUDE_DIR})
    set(CARGO_LIBRARIES ${CARGO_LIBRARY_PYTHON})
endif(NOT CARGO_LIBRARIES OR NOT CARGO_INCLUDE_DIRS)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Cargo DEFAULT_MSG CARGO_LIBRARIES CARGO_INCLUDE_DIRS)

