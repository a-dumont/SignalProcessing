import os
s = os.path.abspath("C:/cygwin64/usr/x86_64-w64-mingw32/sys-root/mingw/bin")

if os.name == "nt" and s not in os.environ["PATH"]:
  #os.environ["PATH"] = s+";"+os.environ["PATH"]
  os.add_dll_directory(s)
  os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

from .libfft import *

try:
    from .libfftcuda import *
    del libfftcuda
except ImportError:
    print("No CUDA support")
except ModuleNotFoundError:
    print("No CUDA support")

del s
del os
del libfft
