#! /bin/bash
#
# Shell scripts to build canny edge detector demo
# - src/main.cpp - src/canny.cpp
# --------------------------------------------------------------
# Created by: Daniel Kurniadi
# Date: at 1 April 2020 (No April Fools)
# Copyright (c) 2020 Daniel Kurniadi. All rights reserved.
# --------------------------------------------------------------

set -e

# --------------------------------------------------------------
# ERROR HANDLING
# --------------------------------------------------------------

cleanup() {
  if [ -f *.o ]; then
    rm *.o
  fi
}

trap cleanup 0

error() {
  local parent_lineno="$1"
  local message="$2"
  local code="${3:-1}"
  if [[ -n "$message" ]] ; then
    echo "Error on or near line ${parent_lineno}: ${message}; exiting with status ${code}"
  else
    echo "Error on or near line ${parent_lineno}; exiting with status ${code}"
  fi
  exit "${code}"
}
trap 'error ${LINENO}' ERR

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

# define directories
PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"

echo ".. creating build/ directory at $PROJ_DIR/build"
cd $PROJ_DIR

# delete and recreate ./build/ directory
if [ -d build ]; then rm -rf build/; fi
mkdir build/

# C++ main file
CPP_MAIN_FILE="${PROJ_DIR}/src/main.cpp"
echo ".. make build from cpp code: $CPP_MAIN_FILE"

# Make build install
make

# cleanup directories from build
if [ -f *.o ]; then
    rm *.o
fi

echo ".. done"

exit 0