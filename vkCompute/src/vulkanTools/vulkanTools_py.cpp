#include "vulkanTools_py.h"

uint32_a PhysicalDeviceInfoPy::getGraphicsFamiliesPy()
{
	return uint32_a
			(
			 PhysicalDeviceInfoPy::getHowmanyGraphicsFamilies(),
			 PhysicalDeviceInfoPy::getGraphicsFamilies()
			);
};

uint32_a PhysicalDeviceInfoPy::getComputeFamiliesPy()
{
	return uint32_a
			(
			 PhysicalDeviceInfoPy::getHowmanyComputeFamilies(),
			 PhysicalDeviceInfoPy::getComputeFamilies()
			);
}

py::list VulkanBasePy::getRequiredLayersPy()
{
    py::list layers;
	std::string str;
	for (uint32_t i = 0; i<getRequiredLayersCount(); i++)
	{
		str = getRequiredLayers()[i];
        layers.append(str);
	}
    return layers;
}

py::list VulkanBasePy::getRequiredExtensionsPy()
{
    py::list extensions;
	std::string str;
	for (uint32_t i = 0; i<getRequiredExtensionsCount(); i++)
	{
		str = getRequiredExtensions()[i];
        extensions.append(str);
	}
    return extensions;
}


void init_vkTools(py::module &m)
{
	// Queue family
	py::class_<QueueFamilyInfoPy>(m,"QueueFamilyInfo")
			.def(py::init())
			.def("hasGraphicsSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasComputeSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasTransferSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasSparseBindingSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasVideoDecodeSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasVideoEncodeSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasOpticalFlowNVRSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("isProtected",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("getIndex",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("getQueueCount",&QueueFamilyInfoPy::hasGraphicsSupport);

	// Physical Device Info
	py::class_<PhysicalDeviceInfoPy>(m,"PhysicalDeviceInfo")
			.def(py::init())
			.def("getGraphicsFamilies",&PhysicalDeviceInfoPy::getGraphicsFamiliesPy)
			.def("getComputeFamilies",&PhysicalDeviceInfoPy::getComputeFamiliesPy)
			.def("getHowmanyExtensions",&PhysicalDeviceInfoPy::getHowmanyExtensions)
			.def("getHowmanyQueueFamilies",&PhysicalDeviceInfoPy::getHowmanyQueueFamilies)
			.def("getHowmanyGraphicsFamilies",&PhysicalDeviceInfoPy::getHowmanyGraphicsFamilies)
			.def("getHowmanyComputeFamilies",&PhysicalDeviceInfoPy::getHowmanyComputeFamilies)
			.def("hasSwapChainSupport",&PhysicalDeviceInfoPy::hasSwapChainSupport)
			.def("printDeviceInfo",&PhysicalDeviceInfoPy::printDeviceInfo);

	// Vulkan base
	py::class_<VulkanBasePy>(m,"VulkanBase")
		.def(py::init([](py::list str_list) {
        // Convert Python list -> vector of strings -> const char**
		//std::string* strings = (std::string*) malloc(n*sizeof(std::string));
		uint32_t n = str_list.size();	
		std::vector<std::string> strings;
        strings.reserve(n);
		const char** ptrs = (const char**) malloc(n*sizeof(const char*));
        for (uint32_t i=0;i<n;i++)
		{
            strings.push_back(str_list[i].cast<std::string>());
            ptrs[i]=strings[i].c_str();
		}

        return std::make_unique<VulkanBasePy>(n,ptrs);}))
			.def("getPhysicalDevicesCount",&VulkanBasePy::getPhysicalDevicesCount)
			.def("getRequiredLayersCount",&VulkanBasePy::getRequiredLayersCount)
			.def("getRequiredLayers",&VulkanBasePy::getRequiredLayersPy)
			.def("getRequiredExtensionsCount",&VulkanBasePy::getRequiredExtensionsCount)
			.def("getRequiredExtensions",&VulkanBasePy::getRequiredExtensionsPy);
}

PYBIND11_MODULE(libvktools, m)
{
	m.doc() = "Everything needed to initialize a Vulkan instance";
	init_vkTools(m);
}

