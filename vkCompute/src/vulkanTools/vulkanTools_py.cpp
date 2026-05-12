#include "vulkanTools_py.h"

void PhysicalDeviceInfoPy::initPy()
{
	uint32_t n = getHowmanyQueueFamilies();
	queueFamiliesInfoPy = (QueueFamilyInfoPy*) malloc(n*sizeof(QueueFamilyInfoPy));
	for(uint32_t i=0;i<n;i++)
	{
		queueFamiliesInfoPy[i].init(getQueueFamilies()[i]);
	}
	isInitPy = true;
}

PhysicalDeviceInfoPy::~PhysicalDeviceInfoPy()
{
	if(isInitPy){free(queueFamiliesInfoPy);}
}

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

py::list PhysicalDeviceInfoPy::getQueueFamiliesInfoPy()
{
	py::list out;
	uint32_t n = getHowmanyQueueFamilies();
	for(uint32_t i=0;i<n;i++)
	{
		out.append(queueFamiliesInfoPy[i]);
	}
	return out;
}

VulkanBasePy::VulkanBasePy(uint32_t nReqLayers, const char** reqLayers) : VulkanBasePy::VulkanBase{nReqLayers,reqLayers}
{
	uint32_t n = getPhysicalDevicesCount();
	physicalDevicesInfoPy = (PhysicalDeviceInfoPy*) malloc(n*sizeof(PhysicalDeviceInfoPy));
	for(uint32_t i=0;i<n;i++)
	{
		physicalDevicesInfoPy[i].init(getPhysicalDevices()[i]);
		physicalDevicesInfoPy[i].initPy();
	}
	isInitPy = true;
}

VulkanBasePy::~VulkanBasePy()
{
	if(isInitPy){free(physicalDevicesInfoPy);}
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

py::list VulkanBasePy::getPhysicalDevicesInfoPy()
{
	py::list out;
	uint32_t n = getPhysicalDevicesCount();
	for(uint32_t i=0;i<n;i++)
	{
		out.append(physicalDevicesInfoPy[i]);
	}
	return out;
}


void init_vkTools(py::module &m)
{
	// Queue family
	py::class_<QueueFamilyInfoPy>(m,"QueueFamilyInfo")
			.def(py::init())
			.def("hasGraphicsSupport",&QueueFamilyInfoPy::hasGraphicsSupport)
			.def("hasComputeSupport",&QueueFamilyInfoPy::hasComputeSupport)
			.def("hasTransferSupport",&QueueFamilyInfoPy::hasTransferSupport)
			.def("hasSparseBindingSupport",&QueueFamilyInfoPy::hasSparseBindingSupport)
			.def("hasVideoDecodeSupport",&QueueFamilyInfoPy::hasVideoDecodeSupport)
			.def("hasVideoEncodeSupport",&QueueFamilyInfoPy::hasVideoEncodeSupport)
			.def("hasOpticalFlowNVRSupport",&QueueFamilyInfoPy::hasOpticalFlowNVRSupport)
			.def("isProtected",&QueueFamilyInfoPy::isProtected)
			.def("getQueueCount",&QueueFamilyInfoPy::getQueueCount);

	// Physical Device Info
	py::class_<PhysicalDeviceInfoPy>(m,"PhysicalDeviceInfo")
			.def(py::init())
			.def("getQueueFamiliesInfo",&PhysicalDeviceInfoPy::getQueueFamiliesInfoPy)
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
			.def("getPhysicalDeviceInfo",&VulkanBasePy::getPhysicalDevicesInfoPy)
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

