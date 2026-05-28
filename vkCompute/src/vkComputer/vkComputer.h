#pragma once

#include "../vulkanTools/vulkanTools.h"

namespace vkComputer
{
enum vkMemcpyFlags
{
	HostToDevice = 0,
	DeviceToHost = 1,
};

class Computer
{
	public:
		//Computer(vkTools::ComputePipeline* pipelineIn, uint32_t invocationSizeIn);
		Computer(vkTools::VulkanBase* vkbaseIn, 
						vkTools::LogicalDevice* logicalDeviceIn,
						uint32_t invocationSizeIn);
		~Computer();
		void recordCommandBuffer(vkTools::ComputePipeline* pipeline, 
						VkCommandBuffer buffer, uint32_t dataLength);
		
		void compute();
		void createDescriptorSetLayout(uint32_t N);
		VkDescriptorSetLayout* getDescriptorSetLayout();

		void vkMemcpy(void* dst, void* src, uint64_t size, uint64_t dstOffset, uint64_t srcOffset,
						vkMemcpyFlags flag);
		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
						VkMemoryPropertyFlags properties,VkBuffer& buffer, 
						VkDeviceMemory& bufferMemory);
		
		void fillBaseWriteDescriptorSet(uint32_t n, VkWriteDescriptorSet* writeDescriptorSet);
		void copyBuffer(VkBuffer src, 
						VkBuffer dst, 
						VkDeviceSize size, 
						uint32_t dstOffset, 
						uint32_t srcOffset, VkQueue queue);
		
		VkCommandBuffer getCommandBuffer();
		uint32_t getInvocationSize();

	private:
		// Vulkan backend
		vkTools::VulkanBase* vkBase;
		vkTools::LogicalDevice* logicalDevice;

		// Workgroup limits
		uint32_t workGroupMaxCount[3];
		uint32_t workGroupMaxSize[3];
		uint32_t maxInvocationSize;
		uint32_t invocationSize;

		// Sync objects
		VkCommandBuffer commandBuffer, memcpyCmdBuffer;
		VkFence computeFence;
		void createSyncObjects();
		void destroySyncObjects();

		// Buffer tools
		void createCommandBuffer();
		//void recordCommandBuffer(VkCommandBuffer buffer, uint32_t dispatchNumber);

		// Bool
		bool descriptorSetLayoutInit = false;

		// Buffers
		uint32_t dataLength = 256;
		
		VkBuffer inputBuffers;
		VkBuffer outputBuffer;
		
		VkDeviceMemory inputMemory;
		VkDeviceMemory outputMemory;

		void* pInputMemory;
		void* pOutputMemory;
		
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkDescriptorPool inOutDescriptorPool;

		void createInOutBuffers();
};
}
