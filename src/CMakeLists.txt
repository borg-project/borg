#
# project basics
#

project(borg)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake CACHE PATH "")

#
# default compiler flags
#

set(CMAKE_C_FLAGS_EXTRA         "-Wall"                              CACHE STRING "")
set(CMAKE_C_FLAGS_DEBUG_EXTRA   "-ggdb"                              CACHE STRING "")
set(CMAKE_CXX_FLAGS_EXTRA       "${CMAKE_C_FLAGS_EXTRA} --std=c++0x" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG_EXTRA "${CMAKE_C_FLAGS_DEBUG_EXTRA}"       CACHE STRING "")

set(CMAKE_C_FLAGS         "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_EXTRA}")
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_DEBUG_EXTRA}")
set(CMAKE_CXX_FLAGS       "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_EXTRA}")

#
# interaction with boost
#

set(Boost_USE_MULTITHREAD ON)
set(Boost_ADDITIONAL_VERSIONS "1.42.0" "1.42")

find_package(Boost "1.42" REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

#
# interaction with cargo
#

find_package(Cargo REQUIRED)

include_directories(${CARGO_INCLUDE_DIRS})

#
# project contents
#

#add_subdirectory(cpp)
add_subdirectory(python)

