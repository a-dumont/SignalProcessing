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

## Copying the repo
Go to your build directory (Ex: Downloads) and git clone the repository:
```console
foo@bar:~$ cd Downloads
foo@bar:~$ git clone https://github.com/a-dumont/SignalProcessing
```
## Build from source
On linux, go to the new SignalProcessing directory and run the installation commands:
```console
foo@bar:~$ bash build.sh
foo@bar:~$ python setup.py install
```
On Windows, as usual, thing are slightly more complicated :
- Running the bash script in cygwin, you may get a "command not found" error. In this case convert the bash file to unix format first
```console
foo@bar:~$ dos2unix build.sh
foo@bar:~$ bash build.sh
```
- Then in you're python environnement (which might not be Cygwin) run the installation
```console
foo@bar:~$ python setup.py install
```
