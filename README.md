# SignalProcessing
C++ methods wrapped with pybind11 for signal processing and whatever else I might need.

# Installation
## Before installing
- Make sure Python and pip are installed on your system.
- On windows it is possible to compile via mingw in cygwin.

# Dependencies
- CMake
- FFTW3
- pybind11
- OpenMP
- minGW (To compile on or for Windows)
- CUDA (Optional)
- cufft (Optional)
- CUDA runtime api (Optional)
- Tested only with g++

## Build from source
Go to your build directory (Ex: Downloads) and git clone the repository:
```console
foo@bar:~$ cd Downloads
foo@bar:~$ git clone https://github.com/a-dumont/SignalProcessing
```

Go to the new SignalProcessing directory and run the installation command:
```console
foo@bar:~$ cd SignalProcessing/
foo@bar:~$ mkdir build && cd build
foo@bar:~$ cmake .. && cmake --build . && cmake --install .
foo@bar:~$ python setup.py install
```

To compile and install with CUDA support for Nvidia GPUs:
```console
foo@bar:~$ cd SignalProcessing/
foo@bar:~$ mkdir build && cd build
foo@bar:~$ ENABLE_CUDA=1 cmake .. && cmake --build . && cmake --install .
foo@bar:~$ python setup.py install
```
