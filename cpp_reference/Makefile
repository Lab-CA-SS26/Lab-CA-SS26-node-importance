# Detect operating system
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    # macOS compilation with Homebrew OpenMP
    CXXFLAGS = -Xpreprocessor -fopenmp -std=c++11 -O3 -Wall -g -Iinclude -I/opt/homebrew/opt/libomp/include
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lm
else
    # Linux compilation
    CXXFLAGS = -fopenmp -std=c++11 -O3 -Wall -g -Iinclude
    LDFLAGS = -lm
endif

build:
	g++ $(CXXFLAGS) main.cpp src/*.cpp $(LDFLAGS) -o kadabra

