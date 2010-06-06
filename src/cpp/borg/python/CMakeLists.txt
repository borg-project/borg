find_package(Numpy REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)

include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    )

add_library(
    borg_python SHARED
    _borg.cc
    abstract_action.cc
    abstract_model.cc
    abstract_outcome.cc
    )

target_link_libraries(
    borg_python
	${CARGO_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )

