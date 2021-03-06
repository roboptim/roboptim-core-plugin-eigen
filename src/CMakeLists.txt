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

# Define the directory where plug-ins will be installed.
IF(ROBOPTIM_CORE_FOUND)
  GET_FILENAME_COMPONENT(RELPLUGINDIR ${ROBOPTIM_CORE_PLUGINDIR} NAME)
ELSE()
  GET_FILENAME_COMPONENT(RELPLUGINDIR ${ROBOPTIM_CORE_DEBUG_PLUGINDIR} NAME)
ENDIF(ROBOPTIM_CORE_FOUND)
SET(PLUGINDIR ${CMAKE_INSTALL_LIBDIR}/${RELPLUGINDIR})

MACRO(EIGEN_PLUGIN NAME)
  ADD_LIBRARY(roboptim-core-plugin-${NAME} MODULE ${NAME}.cc ${HEADERS})
  PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-${NAME} roboptim-core)
  SET_TARGET_PROPERTIES(roboptim-core-plugin-${NAME} PROPERTIES
    PREFIX "")
  IF(NOT APPLE)
    SET_TARGET_PROPERTIES(roboptim-core-plugin-${NAME}
      PROPERTIES VERSION 3.2.0 SOVERSION 3)
  ENDIF()
  INSTALL(TARGETS roboptim-core-plugin-${NAME}
    DESTINATION ${PLUGINDIR})
ENDMACRO()

# Levenberg-Marquardt solver
EIGEN_PLUGIN("eigen-levenberg-marquardt")
