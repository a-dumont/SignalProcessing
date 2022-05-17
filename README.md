# SignalProcessing
C++ methods wrapped with pybind11 for signal processing and whatever else I might need.

# Installation
## Before installing
- Make sure Python and pip are installed on your system.
- On windows it is possible to compile via mingw in wsl2 or cygwin.

# Dependencies
- FFTW3
- pybind11
- OpenMP
- mingw (To compile on or for Windows)
- Tested only with g++

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

# Contents so far
- FFT
	- Wrappers around FFTW3 to easily interface with python.
- Correlations
	- Uses FFT to compute correlation in the frequency domain.
	- TODO add time correlations/convolutions.
	- TODO wrap ifft to give a time result.
- Histograms
	- 1D or 2D Histograms.
- Math
	- General purpose arithmetic.
	- Statistics, mean, variance, skewness.
	- Numerical gradient.s
