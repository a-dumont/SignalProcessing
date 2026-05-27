import os
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
    from ..vkCompute import _vulkanBackend
    logicalDev = _vulkanBackend.getLogicalDevices()[0]
    import SignalProcessing.math.libvkmath as _libvkmath
    _pipelines = [logicalDev.createComputePipeline("/home/alex/Codes/SignalProcessing/vkMath/bin/Shaders/1d_uint_vAdd_%i.spv"%(1<<i)) for i in range(2,11)]
    vkAdd = lambda x,y: _libvkmath.vkAdd(_vulkanBackend.getComputers()[-1],_pipelines[-1],x,y)
    #functions = inspect.getmembers(libvkmath,inspect.isbuiltin)
    #for i in range(len(functions)):
        #setattr(self,functions[i],)

except:
    print("No vulkan support.")

del s
del os
del inspect
del libmath
