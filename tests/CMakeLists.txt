# Copyright (c) 2013 CNRS
# Authors: Benjamin Chretien
#
#
# This file is part of roboptim-core-plugin-eigen
# roboptim-core-plugin-eigen is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# roboptim-core-plugin-eigen is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Lesser Public License for more details.  You should have
# received a copy of the GNU Lesser General Public License along with
# roboptim-core-plugin-eigen  If not, see
# <http://www.gnu.org/licenses/>.

SET(PLUGIN_PATH "${CMAKE_BINARY_DIR}/src")

SET(SOLVER_NAME "eigen-levenberg-marquardt")
SET(FUNCTION_TYPE ::roboptim::EigenMatrixDense)
SET(COST_FUNCTION_TYPE ::roboptim::GenericFunction)
SET(CONSTRAINT_TYPE_1 False)
SET(CONSTRAINT_TYPE_2 False)
SET(PROGRAM_SUFFIX "")

# Make sure local headers are found
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/tests)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/tests)

# Add Boost path to include directories.
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

# Make Boost.Test generates the main function in test cases.
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

SET(COMMON_HEADERS
  ${HEADERS}
  shared-tests/common.hh
  )

ADD_EXECUTABLE(distance-to-sphere
  ${CMAKE_SOURCE_DIR}/tests/shared-tests/distance-to-sphere.cc
  ${COMMON_HEADERS}
  ${CMAKE_SOURCE_DIR}/src/eigen-levenberg-marquardt.cc
  )

SET_TARGET_PROPERTIES(distance-to-sphere
    PROPERTIES COMPILE_FLAGS
    "-DSOLVER_NAME='\"${SOLVER_NAME}\"' -DPLUGIN_PATH='\"${PLUGIN_PATH}\"' -DFUNCTION_TYPE=${FUNCTION_TYPE} -DCOST_FUNCTION_TYPE=${COST_FUNCTION_TYPE} ${CONSTRAINT_TYPE_1_STR} ${CONSTRAINT_TYPE_2_STR} -DTESTS_DATA_DIR='\"${CMAKE_CURRENT_SOURCE_DIR}\"' -DLOG_FILENAME='\"${NAME}${PROGRAM_SUFFIX}.log\"' ")
PKG_CONFIG_USE_DEPENDENCY(distance-to-sphere roboptim-core)
TARGET_LINK_LIBRARIES(distance-to-sphere ${Boost_LIBRARIES})
ADD_TEST(distance-to-sphere ${RUNTIME_OUTPUT_DIRECTORY}/distance-to-sphere)



