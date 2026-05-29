#include "vulkanTools_py.h"
#include "vulkanTools.h"
#include <cstring>
#include <memory>
#include <stdexcept>

void PhysicalDeviceInfoPy::initPy()
{
	uint32_t n = getHowmanyQueueFamilies();
	queueFamiliesInfoPy = (QueueFamilyInfoPy*) malloc(n*sizeof(QueueFamilyInfoPy));
	isInitPy = true;
	for(uint32_t i=0;i<n;i++)
	{
		queueFamiliesInfoPy[i].init(getQueueFamilies()[i]);
	}
}

PhysicalDeviceInfoPy::~PhysicalDeviceInfoPy()
{
	if(isInitPy){free(queueFamiliesInfoPy);isInitPy=false;}
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
		out.append(&queueFamiliesInfoPy[i]);
	}
	return out;
}

std::string PhysicalDeviceInfoPy::getDeviceName()
{
	return std::string(getProperties().deviceName);
}

py::dict PhysicalDeviceInfoPy::getPhysicalDeviceLimits()
{
	py::dict dict;
	VkPhysicalDeviceLimits limits = getProperties().limits;

	dict["maxImageDimension1D"] = limits.maxImageDimension1D;
	dict["maxImageDimension2D"] = limits.maxImageDimension2D;
	dict["maxImageDimension3D"] = limits.maxImageDimension3D;
	dict["maxImageDimensionCube"] = limits.maxImageDimensionCube;
	dict["maxImageArrayLayers"] = limits.maxImageArrayLayers;
	dict["maxTexelBufferElements"] = limits.maxTexelBufferElements;
	dict["maxUniformBufferRange"] = limits.maxUniformBufferRange;
	dict["maxStorageBufferRange"] = limits.maxStorageBufferRange;
	dict["maxPushConstantsSize"] = limits.maxPushConstantsSize;
	dict["maxMemoryAllocationCount"] = limits.maxMemoryAllocationCount;
	dict["maxSamplerAllocationCount"] = limits.maxSamplerAllocationCount;
	dict["bufferImageGranularity"] = (uint64_t) limits.bufferImageGranularity;
	dict["sparseAddressSpaceSize"] = (uint64_t) limits.sparseAddressSpaceSize;
	dict["maxBoundDescriptorSet"] = limits.maxBoundDescriptorSets;
	dict["maxPerStageDescriptorSampler"] = limits.maxPerStageDescriptorSamplers;
	dict["maxPerStageDescriptorUniformBuffers"] = limits.maxPerStageDescriptorUniformBuffers;
	dict["maxPerStageDescriptorStorageBuffers"] = limits.maxPerStageDescriptorStorageBuffers;
	dict["maxPerStageDescriptorSampledImages"] = limits.maxPerStageDescriptorSampledImages;
	dict["maxPerStageDescriptorStorageImages"] = limits.maxPerStageDescriptorStorageImages;
	dict["maxPerStageDescriptorInputAttachements"] = limits.maxPerStageDescriptorInputAttachments;
	dict["maxPerStageResources"] = limits.maxPerStageResources;
	dict["maxDescriptorSetSamplers"] = limits.maxDescriptorSetSamplers;
	dict["maxDescriptorSetUniformBuffers"] = limits.maxDescriptorSetUniformBuffers;
	dict["maxDescriptorSetUniformBuffersDynamic"] = limits.maxDescriptorSetUniformBuffersDynamic;
	dict["maxDescriptorSetStorageBuffers"] = limits.maxDescriptorSetStorageBuffers;
	dict["maxDescriptorSetStorageBuffersDynamic"] = limits.maxDescriptorSetStorageBuffersDynamic;
	dict["maxDescriptorSetSampledImages"] = limits.maxDescriptorSetSampledImages;
	dict["maxDescriptorSetStorageImages"] = limits.maxDescriptorSetStorageImages;
	dict["maxDescriptorSetInputAttachements"] = limits.maxDescriptorSetInputAttachments;
	dict["maxVertexInputAttributes"] = limits.maxVertexInputAttributes;
	dict["maxVertexInputBindings"] = limits.maxVertexInputBindings;
	dict["maxVertexInputAttributeOffset"] = limits.maxVertexInputAttributeOffset;
	dict["maxVertexInputBindingStride"] = limits.maxVertexInputBindingStride;
	dict["maxVertexOutputComponents"] = limits.maxVertexOutputComponents;
	dict["maxTessellationGenerationLevel"] = limits.maxTessellationGenerationLevel;
	dict["maxTessellationPatchSize"] = limits.maxTessellationPatchSize;
	dict["maxTessellationControlPerVertexInputComponents"] = 
			limits.maxTessellationControlPerVertexInputComponents;
	dict["maxTessellationControlPerVertexOutputComponents"] = 
			limits.maxTessellationControlPerVertexOutputComponents;
	dict["maxTessellationControlPerPatchOutputComponents"] = 
			limits.maxTessellationControlPerPatchOutputComponents;
	dict["maxTessellationControlTotalOutputComponents"] = 
			limits.maxTessellationControlTotalOutputComponents;
	dict["maxTessellationEvaluationInputComponents"] = 
			limits.maxTessellationEvaluationInputComponents;
	dict["maxTessellationEvaluationOutputComponents"] = 
			limits.maxTessellationEvaluationOutputComponents;
	dict["maxGeometryShaderInvocations"] = limits.maxGeometryShaderInvocations;
	dict["maxGeometryInputComponents"] = limits.maxGeometryInputComponents;
	dict["maxGeometryOutputComponents"] = limits.maxGeometryOutputComponents;
	dict["maxGeometryOutputVertices"] = limits.maxGeometryOutputVertices;
	dict["maxGeometryTotalOutputComponents"] = limits.maxGeometryTotalOutputComponents;
	dict["maxFragmentInputComponents"] = limits.maxFragmentInputComponents;
	dict["maxFragmentOutputAttachements"] = limits.maxFragmentOutputAttachments;
	dict["maxFragmentDualSrcAttachements"] = limits.maxFragmentDualSrcAttachments;
	dict["maxFragmentCombinedOutputRessources"] = limits.maxFragmentCombinedOutputResources;
	dict["maxComputeSharedMemorySize"] = limits.maxComputeSharedMemorySize;
	dict["maxComputeWorkGroupCount"] = 
			uint32_a({3},{sizeof(uint32_t)},limits.maxComputeWorkGroupCount);
	dict["maxComputeWorkGroupInvocations"] = limits.maxComputeWorkGroupInvocations;
	dict["maxComputeWorkGroupSize"] =
			uint32_a({3},{sizeof(uint32_t)},limits.maxComputeWorkGroupSize);
	dict["subPixelPrecisionBits"] = limits.subPixelPrecisionBits;
	dict["subTexelPrecisionBits"] = limits.subTexelPrecisionBits;
	dict["mipmapPrecisionBits"] = limits.mipmapPrecisionBits;
	dict["maxDrawIndexedIndexValue"] = limits.maxDrawIndexedIndexValue;
	dict["maxDrawIndirectCount"] = limits.maxDrawIndirectCount;
	dict["maxSamplerLodBias"] = limits.maxSamplerLodBias;
	dict["maxSamplerAnisotropy"] = limits.maxSamplerAnisotropy;
	dict["maxViewports"] = limits.maxViewports;
	dict["maxViewportDimensions"] = uint32_a({2},{sizeof(uint32_t)},limits.maxViewportDimensions);
	dict["viewportBoundsRange"] = float32_a({2},{sizeof(float)},limits.viewportBoundsRange);
	dict["viewportSubpixelBits"] = limits.viewportSubPixelBits;
	dict["minMemoryMapAlignment"] = limits.minMemoryMapAlignment;
	dict["minTexelBufferOffsetAlignment"] = (uint64_t) limits.minTexelBufferOffsetAlignment;
	dict["minUniformBufferOffsetAlignment"] = (uint64_t) limits.minUniformBufferOffsetAlignment;
	dict["minStorageBufferOffsetAlignment"] = (uint64_t) limits.minStorageBufferOffsetAlignment;
	dict["minTexelOffset"] = limits.minTexelOffset;
	dict["maxTexelOffset"] = limits.maxTexelOffset;
	dict["minTexelGatherOffset"] = limits.minTexelGatherOffset;
	dict["maxTexelGatherOffset"] = limits.maxTexelGatherOffset;
	dict["minInterpolationOffset"] = limits.minInterpolationOffset;
	dict["maxInterpolationOffset"] = limits.maxInterpolationOffset;
	dict["subPixelInterpolationOffsetBits"] = limits.subPixelInterpolationOffsetBits;
	dict["maxFramebufferWidth"] = limits.maxFramebufferWidth;
	dict["maxFramebufferHeight"] = limits.maxFramebufferHeight;
	dict["maxFramebufferLayers"] = limits.maxFramebufferLayers;
	dict["framebufferColorSampleCounts"] = (uint32_t) limits.framebufferColorSampleCounts;
	dict["framebufferDepthSampleCounts"] = (uint32_t) limits.framebufferDepthSampleCounts;
	dict["framebufferStencilSampleCounts"] = (uint32_t) limits.framebufferStencilSampleCounts;
	dict["framebufferNoAttachmentsSampleCounts"] = 
			(uint32_t) limits.framebufferNoAttachmentsSampleCounts;
	dict["maxColorAttachments"] = limits.maxColorAttachments;
	dict["sampledImageColorSampleCounts"] = (uint32_t) limits.sampledImageColorSampleCounts;
	dict["sampledImageIntegerSampleCounts"] = (uint32_t) limits.sampledImageIntegerSampleCounts;
	dict["sampledImageDepthSampleCounts"] = (uint32_t) limits.sampledImageDepthSampleCounts;
	dict["sampledImageStencilSampleCounts"] = (uint32_t) limits.sampledImageStencilSampleCounts;
	dict["storageImageSampleCounts"] = (uint32_t) limits.storageImageSampleCounts;
	dict["maxSampleMaskWords"] = limits.maxSampleMaskWords;
	dict["timestampComputeAndGraphics"] = (bool) limits.timestampComputeAndGraphics;
	dict["timestampPeriod"] = limits.timestampPeriod;
	dict["maxClipDistances"] = limits.maxClipDistances;
	dict["maxCullDistances"] = limits.maxCullDistances;
	dict["maxCombinedClipAndCullDistances"] = limits.maxCombinedClipAndCullDistances;
	dict["discreteQueuePriorities"] = limits.discreteQueuePriorities;
	dict["pointSizeRange"] = float32_a({2},{sizeof(float)},limits.pointSizeRange);
	dict["lineWidthRange"] = float32_a({2},{sizeof(float)},limits.lineWidthRange);
	dict["pointSizeGranularity"] = limits.pointSizeGranularity;
	dict["lineWidthGranularity"] = limits.lineWidthGranularity;
	dict["strictLines"] = (bool) limits.strictLines;
	dict["standardSampleLocations"] = (bool) limits.standardSampleLocations;
	dict["optimalBufferCopyOffsetAlignment"] = (uint64_t) limits.optimalBufferCopyOffsetAlignment;
	dict["optimalBufferCopyRowPitchAlignment"] = 
			(uint64_t) limits.optimalBufferCopyRowPitchAlignment;
	dict["nonCoherentAtomSize"] = (uint64_t) limits.nonCoherentAtomSize;

	return dict;
}

py::dict PhysicalDeviceInfoPy::getPhysicalDeviceProperties()
{
	py::dict dict;
	VkPhysicalDeviceProperties prop = getProperties();

	dict["apiVersion"] = py::dict(
					py::arg("Variant") = VK_API_VERSION_VARIANT(prop.apiVersion),
					py::arg("Major") = VK_API_VERSION_MAJOR(prop.apiVersion),
					py::arg("Minor") = VK_API_VERSION_MINOR(prop.apiVersion),
					py::arg("Patch") = VK_API_VERSION_PATCH(prop.apiVersion));
	dict["driverVersion"] = prop.driverVersion;
	dict["vendorId"] = prop.vendorID;
	dict["deviceId"] = prop.deviceID;
	dict["deviceType"] = string_VkPhysicalDeviceType(prop.deviceType);
	dict["deviceName"] = std::string(prop.deviceName);
	dict["pipelineCacheUUID"] = uint8_a({VK_UUID_SIZE},{sizeof(uint8_t)},prop.pipelineCacheUUID);
	dict["limits"] = getPhysicalDeviceLimits();

	return dict;
}

ComputePipelinePy& ComputePipelinePy::operator=(ComputePipelinePy&& existingInstance) noexcept
{
	if(this != &existingInstance)
	{
		// base members
		std::swap(
			static_cast<vkTools::ComputePipeline&>(*this),
            static_cast<vkTools::ComputePipeline&>(existingInstance)
        );
	}
	return *this;
}

LogicalDevicePy::~LogicalDevicePy()
{
	while(howmanyPipelines>0){destroyComputePipeline(pipelinesMap.rbegin()->first);}
	if(pipelinesInit){free(pipelines);}
   	pipelinesMap.clear();	
}

LogicalDevicePy& LogicalDevicePy::operator=(LogicalDevicePy&& existingInstance) noexcept
{
	if(this != &existingInstance)
	{
		// base members
		std::swap(
			static_cast<vkTools::LogicalDevice&>(*this),
            static_cast<vkTools::LogicalDevice&>(existingInstance)
        );
	}
	return *this;
}

/*
std::unique_ptr<ComputePipelinePy> LogicalDevicePy::createComputePipeline(std::string shaderFile)
{
	return std::make_unique<ComputePipelinePy>(this,shaderFile);
}
*/

void LogicalDevicePy::createComputePipeline(std::string shaderFile)
{
	
	if(pipelinesInit == false)
	{
		pipelines = (ComputePipelinePy*) 
				malloc((howmanyPipelines+1)*sizeof(ComputePipelinePy));
		pipelinesInit = true;
	}
	else
	{
		pipelines = (ComputePipelinePy*) 
				realloc((void*)pipelines,(howmanyPipelines+1)*sizeof(ComputePipelinePy));
	}
	new (&pipelines[howmanyPipelines]) ComputePipelinePy(this,shaderFile);
	pipelinesMap[shaderFile] = howmanyPipelines;
	howmanyPipelines += 1;
}

void LogicalDevicePy::destroyComputePipeline(std::string shaderFile)
{
	if(howmanyPipelines == 0){}
	else
	{
		uint32_t devIndex = pipelinesMap[shaderFile];
		pipelines[devIndex].~ComputePipelinePy();
		for(uint32_t i=devIndex;i<howmanyPipelines-1;i++)
		{
			pipelines[i] = std::move(pipelines[i+1]);
			pipelinesMap[shaderFile] = i;
		}
		pipelines = (ComputePipelinePy*) 
				realloc((void*) pipelines,(howmanyPipelines-1)*sizeof(ComputePipelinePy));
		howmanyPipelines -= 1;
		pipelinesMap.erase(shaderFile);
	}
}

py::dict LogicalDevicePy::getComputePipelines()
{
	py::dict out;

	std::map<std::string,uint32_t>::iterator it;
	for(it=pipelinesMap.begin();it!=pipelinesMap.end();++it)
	{
		out[py::cast(it->first)] = &      pipelines[it->second];
	}

	return out;
}

std::string LogicalDevicePy::getPhysicalDeviceName()
{
	return std::string(getPhysicalDeviceInfo()->getProperties().deviceName);
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
	
	while(howmanyComputers>0){destroyComputer(howmanyComputers-1);}
	if(computersInit){free(computers);}
	
	while(howmanyLogicalDevices>0){destroyLogicalDevice(howmanyLogicalDevices-1);}
	if(logicalDevicesInit){free(logicalDevices);}
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
		out.append(&physicalDevicesInfoPy[i]);
	}
	return out;
}

py::list VulkanBasePy::getLogicalDevices()
{
	py::list out;
	for(uint32_t i=0;i<howmanyLogicalDevices;i++)
	{
		out.append(&logicalDevices[i]);
	}
	return out;
}

py::list VulkanBasePy::getComputers()
{
	py::list out;
	for(uint32_t i=0;i<howmanyComputers;i++)
	{
		out.append(&computers[i]);
	}
	return out;
}

void VulkanBasePy::createLogicalDevice(uint32_t pDevIndex, uint32_t usageFlags)
{
	if(logicalDevicesInit == false)
	{
		logicalDevices = (LogicalDevicePy*) 
				malloc((howmanyLogicalDevices+1)*sizeof(LogicalDevicePy));
		logicalDevicesInit = true;
	}
	else
	{
		logicalDevices = (LogicalDevicePy*) 
				realloc((void*)logicalDevices,(howmanyLogicalDevices+1)*sizeof(LogicalDevicePy));
	}
	new (&logicalDevices[howmanyLogicalDevices]) LogicalDevicePy(this,pDevIndex, usageFlags);
	howmanyLogicalDevices += 1;
}

void VulkanBasePy::destroyLogicalDevice(uint32_t devIndex)
{
	if(howmanyLogicalDevices == 0){}
	else
	{
		logicalDevices[devIndex].~LogicalDevicePy();
		for(uint32_t i=devIndex;i<howmanyLogicalDevices-1;i++)
		{
			logicalDevices[i] = std::move(logicalDevices[i+1]);
		}
		logicalDevices = (LogicalDevicePy*) 
				realloc((void*) logicalDevices,(howmanyLogicalDevices-1)*sizeof(LogicalDevicePy));
		howmanyLogicalDevices -= 1;
	}
}

void VulkanBasePy::createComputer(uint32_t size, uint32_t logicalDevIdx)
{
	if(howmanyLogicalDevices==0){throw std::runtime_error("Must have logical devices");}
	if(computersInit == false)
	{
		computers = (vkComputer::Computer*) 
				malloc((howmanyComputers+1)*sizeof(vkComputer::Computer));
		computersInit = true;
	}
	else
	{
		computers = (vkComputer::Computer*) 
				realloc((void*)computers,(howmanyComputers+1)*sizeof(vkComputer::Computer));
	}
	new (&computers[howmanyComputers]) 
			vkComputer::Computer(this,&logicalDevices[logicalDevIdx],size);
	howmanyComputers += 1;
}

void VulkanBasePy::destroyComputer(uint32_t index)
{
	if(howmanyComputers == 0){}
	else
	{
		computers[index].~Computer();
		for(uint32_t i=index;i<howmanyComputers-1;i++)
		{
			computers[i] = computers[i+1];
		}
		computers = (vkComputer::Computer*) 
				realloc((void*) computers,(howmanyComputers-1)*sizeof(vkComputer::Computer));
		howmanyComputers -= 1;
	}
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
			.def("getDeviceName",&PhysicalDeviceInfoPy::getDeviceName)
			.def("getPhysicalDeviceLimits",&PhysicalDeviceInfoPy::getPhysicalDeviceLimits)
			.def("getPhysicalDeviceProperties",&PhysicalDeviceInfoPy::getPhysicalDeviceProperties)
			.def("printDeviceInfo",&PhysicalDeviceInfoPy::printDeviceInfo);

	// Vulkan base
	py::class_<VulkanBasePy,std::unique_ptr<VulkanBasePy>>(m,"VulkanBase")
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
			.def("createLogicalDevice",&VulkanBasePy::createLogicalDevice)
			.def("destroyLogicalDevices",&VulkanBasePy::destroyLogicalDevice)
			.def("createComputer",&VulkanBasePy::createComputer)
			.def("destroyComputer",&VulkanBasePy::destroyComputer)
			.def("getLogicalDevices",&VulkanBasePy::getLogicalDevices,
							py::return_value_policy::reference)
			.def("getComputers",&VulkanBasePy::getComputers,py::return_value_policy::reference)
			.def("getPhysicalDevicesCount",&VulkanBasePy::getPhysicalDevicesCount)
			.def("getPhysicalDeviceInfo",&VulkanBasePy::getPhysicalDevicesInfoPy)
			.def("getRequiredLayersCount",&VulkanBasePy::getRequiredLayersCount)
			.def("getRequiredLayers",&VulkanBasePy::getRequiredLayersPy)
			.def("getRequiredExtensionsCount",&VulkanBasePy::getRequiredExtensionsCount)
			.def("getRequiredExtensions",&VulkanBasePy::getRequiredExtensionsPy);

	// Compute pipeline
	py::class_<ComputePipelinePy>(m,"ComputePipeline")
			.def("recreatePipeline",&ComputePipelinePy::recreatePipeline);

	// Logical Device
	py::class_<LogicalDevicePy>(m,"LogicalDevice")
			.def(py::init<VulkanBasePy*,uint32_t,uint32_t>())
			.def("getPhysicalDeviceName",&LogicalDevicePy::getPhysicalDeviceName)
			.def("createComputePipeline",&LogicalDevicePy::createComputePipeline)
			.def("destroyComputePipeline",&LogicalDevicePy::destroyComputePipeline)
			.def("getComputePipelines",&LogicalDevicePy::getComputePipelines)
			.def("getUsageFlags",&LogicalDevicePy::getUsageFlags);

	// Computer
	py::class_<vkComputer::Computer>(m,"vulkanComputer");
}

PYBIND11_MODULE(libvktools, m)
{
	m.doc() = "Everything needed to initialize a Vulkan instance";
	init_vkTools(m);
}

