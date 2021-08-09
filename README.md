# SignalProcessing
C++ methods wrapped with pybind11 for signal processing.

# Installation
## Before installing
- Make sure Python and pip are installed on your system
- On windows the use of cygwin is recommended

# Dependencies
- fftw3
- pybind11
- mingw (To compile on or for Windows)

## Build from source
Go to your build directory (Ex: Downloads) and git clone the repository:
```console
foo@bar:~$ cd Downloads
foo@bar:~$ git clone https://github.com/a-dumont/SignalProcessing
```

Go to the new SignalProcessing directory and run the installation commands:
```console
foo@bar:~$ bash build.sh
foo@bar:~$ python setup.py install
```
