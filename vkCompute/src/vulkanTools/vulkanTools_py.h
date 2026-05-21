#pragma once
#include "vulkanTools.h"
#include <pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

// uint arrays
typedef py::array_t<uint64_t,py::array::c_style> uint64_a;
typedef py::array_t<uint32_t,py::array::c_style> uint32_a;
typedef py::array_t<uint16_t,py::array::c_style> uint16_a;
typedef py::array_t<uint8_t,py::array::c_style> uint8_a;

// Float arrays
typedef py::array_t<float,py::array::c_style> float32_a;
typedef py::array_t<double,py::array::c_style> float64_a;

// String array
typedef py::array_t<std::string,py::array::c_style> str_a;

// Classes
class QueueFamilyInfoPy : public vkTools::QueueFamilyInfo
{
	public:

	private:
};

class PhysicalDeviceInfoPy: public vkTools::PhysicalDeviceInfo
{
	public:
	using vkTools::PhysicalDeviceInfo::PhysicalDeviceInfo;
	~PhysicalDeviceInfoPy();
	void initPy();
	
	uint32_a getGraphicsFamiliesPy();
	uint32_a getComputeFamiliesPy();
	
	py::list getQueueFamiliesInfoPy();
	std::string getDeviceName();
	py::dict getPhysicalDeviceLimits();
	py::dict getPhysicalDeviceProperties();

	private:
	bool isInitPy = false;
	QueueFamilyInfoPy* queueFamiliesInfoPy;
};

class ComputePipelinePy: public vkTools::ComputePipeline
{
	public:
	using vkTools::ComputePipeline::ComputePipeline;

	private:
};

class LogicalDevicePy: public vkTools::LogicalDevice
{
	public:
	using vkTools::LogicalDevice::LogicalDevice;
	~LogicalDevicePy();
	std::string getPhysicalDeviceName();
	void createComputePipeline(const char* shaderFile);
	void destroyComputePipeline(uint32_t pipelineIndex);

	private:
	ComputePipelinePy* pipelines;
	uint32_t howmanyPipelines=0;
	bool pipelinesInit = false;
};

class VulkanBasePy: public vkTools::VulkanBase
{
	public:	
	VulkanBasePy(uint32_t nReqLayers, const char** reqLayers);
	~VulkanBasePy();

	void createLogicalDevice(uint32_t pDevIndex, uint32_t usageFlags);
	void destroyLogicalDevice(uint32_t devIndex);

	py::list getRequiredLayersPy();
	py::list getRequiredExtensionsPy();
	py::list getPhysicalDevicesInfoPy();
	py::list getLogicalDevices();

	private:
	bool isInitPy = false;
	PhysicalDeviceInfoPy* physicalDevicesInfoPy;
	
	LogicalDevicePy* logicalDevices;
	uint32_t howmanyLogicalDevices = 0;
	bool logicalDevicesInit = false;
};
