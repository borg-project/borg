#
# project basics
#

project(borg)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/cmake CACHE PATH "")

#
# project contents
#

add_subdirectory(src)

