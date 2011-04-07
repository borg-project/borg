#
# configure python dependencies
#

find_package(Numpy REQUIRED)
find_package(PythonHeaders REQUIRED)

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS} # should be last
    )

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

