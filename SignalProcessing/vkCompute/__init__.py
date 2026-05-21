import os
s = os.path.abspath("C:/cygwin64/usr/x86_64-w64-mingw32/sys-root/mingw/bin")

if os.name == "nt" and s not in os.environ["PATH"]:
  #os.environ["PATH"] = s+";"+os.environ["PATH"]
  os.add_dll_directory(s)
  os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

try:
    from .libvktools import *
    _vulkanBackend = VulkanBase(["VK_LAYER_KHRONOS_validation"])
    physicalDevices = _vulkanBackend.getPhysicalDeviceInfo()
    assert len(physicalDevices) > 0
    for i in range(len(physicalDevices)):
        if(physicalDevices[i].getPhysicalDeviceProperties()['deviceType'] == 'VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU'):
            _vulkanBackend.createLogicalDevice(i,2|4)
            break
    if len(_vulkanBackend.getLogicalDevices())==0:
        _vulkanBackend.createLogicalDevice(0,2|4)

    del libvktools
    del i
except AssertionError:
    print("No Vulkan device found.")
except ImportError:
    print("No Vulkan support")
except ModuleNotFoundError:
    print("No Vulkan support")

del s
del os
del physicalDevices
