import os
import sys
import inspect
s = os.path.abspath("C:/cygwin64/usr/x86_64-w64-mingw32/sys-root/mingw/bin")

if os.name == "nt" and s not in os.environ["PATH"]:
  #os.environ["PATH"] = s+";"+os.environ["PATH"]
  os.add_dll_directory(s)
  os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

from .libmath import *

# CUDA
try:
    from .libmathcuda import *
    del libmathcuda
except ImportError:
    print("No CUDA support")
except ModuleNotFoundError:
    print("No CUDA support")

# Vulkan
try:
    import numpy as _np
    from ..vkCompute import _vulkanBackend
    _logicalDev = _vulkanBackend.getLogicalDevices()[0]
    import SignalProcessing.math.libvkmath as _libvkmath
    _path = os.path.abspath(os.path.dirname(__file__))+"/shaders/"
    _shaders = os.listdir(_path)
    _shaders = [os.path.join(_path,f) for f in _shaders if os.path.isfile(os.path.join(_path, f))]
    _shaders.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for i in list(dict.fromkeys([int(i.split("_")[-1].split(".")[0]) for i in _shaders])):
        _vulkanBackend.createComputer(i,0)
    del i

    _funcs = dict(inspect.getmembers(_libvkmath,inspect.isbuiltin))
    for _shader in _shaders:
        _logicalDev.createComputePipeline(_shader)
        _i = os.path.basename(_shader).split("_")[-1].split(".")[0]
        _name = os.path.basename(_shader).split("_")[0]
        setattr(sys.modules[__name__],_name+"_"+_i,lambda *args: _funcs[_name](_vulkanBackend.getComputers()[int(_np.log2(int(_i)))-4],_logicalDev.getComputePipelines()[_shader],*args))
        setattr(getattr(sys.modules[__name__],_name+"_"+_i),'__doc__',getattr(_funcs[_name],'__doc__'))

except ImportError:
    print("No vulkan support.")

del s
del os
del sys
del inspect
del libmath
