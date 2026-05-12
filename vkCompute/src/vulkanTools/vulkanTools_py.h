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

	private:
	bool isInitPy = false;
	QueueFamilyInfoPy* queueFamiliesInfoPy;
};

class VulkanBasePy: public vkTools::VulkanBase
{
	public:	
	VulkanBasePy(uint32_t nReqLayers, const char** reqLayers);
	~VulkanBasePy();
	py::list getRequiredLayersPy();
	py::list getRequiredExtensionsPy();
	py::list getPhysicalDevicesInfoPy();

	private:
	bool isInitPy = false;
	PhysicalDeviceInfoPy* physicalDevicesInfoPy;
};

class LogicalDevicePy: public vkTools::LogicalDevice
{
	public:

	private:
};

class ComputePipelinePy: public vkTools::ComputePipeline
{
	public:

	private:
};
