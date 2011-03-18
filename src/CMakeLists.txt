#
# default compiler flags
#

set(CMAKE_C_FLAGS_EXTRA         ""                                   CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG_EXTRA   "-fno-strict-aliasing -ggdb -O2"     CACHE STRING "")
set(CMAKE_CXX_FLAGS_EXTRA       "${CMAKE_C_FLAGS_EXTRA} --std=c++0x" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG_EXTRA "${CMAKE_C_FLAGS_DEBUG_EXTRA}"       CACHE STRING "")

set(CMAKE_C_FLAGS         "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_EXTRA}")
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_DEBUG_EXTRA}")
set(CMAKE_CXX_FLAGS       "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA}")

#
# descend
#

add_subdirectory(python)

