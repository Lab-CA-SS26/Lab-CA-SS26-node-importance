#! /bin/sh
./checkout_submodules.sh && \
./copy_files.sh && \
./compile.sh && \
./run_all_exact.sh && \
./run_all_bavarian.sh && \
./generate_all_figures.sh
