#! /bin/sh
rm -rf build && \
mkdir build && \
cd build && \
cmake -DCMAKE_INSTALL_PREFIX=.. .. && \
make -j4 install && \
cd ../
