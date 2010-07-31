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
# configure cython
#

find_package(Cython REQUIRED)

include(UseCython)

#
# descend
#

add_subdirectory(borg)

