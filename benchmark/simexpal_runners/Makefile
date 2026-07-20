# Detect operating system
UNAME_S := $(shell uname -s)

CPP_REF_DIR = ../../cpp_reference
CXXFLAGS = -std=c++11 -O3 -Wall -g -I$(CPP_REF_DIR)/include

ifeq ($(UNAME_S), Darwin)
    # macOS compilation with Homebrew OpenMP
    CXXFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lm
else
    # Linux compilation
    CXXFLAGS += -fopenmp
    LDFLAGS = -lm
endif

build:
	g++ $(CXXFLAGS) run_experiments.cpp $(CPP_REF_DIR)/src/*.cpp $(LDFLAGS) -o run_experiments_cpp
