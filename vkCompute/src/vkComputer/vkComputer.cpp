#include "vkComputer.h"
#include <stdexcept>

using namespace vkComputer;

Computer::Computer(
				vkTools::VulkanBase* vkBaseIn, 
				vkTools::LogicalDevice* logicalDeviceIn, 
				uint32_t invocationSizeIn)
{
	// Set devices	
	vkBase = vkBaseIn;
	logicalDevice = logicalDeviceIn;
	VkPhysicalDeviceLimits limits = logicalDevice->getPhysicalDeviceInfo()->getProperties().limits;

	// Set limits
	workGroupMaxCount[0] = limits.maxComputeWorkGroupCount[0];
	workGroupMaxCount[1] = limits.maxComputeWorkGroupCount[1];
	workGroupMaxCount[2] = limits.maxComputeWorkGroupCount[2];
	
	workGroupMaxSize[0] = limits.maxComputeWorkGroupSize[0];
	workGroupMaxSize[1] = limits.maxComputeWorkGroupSize[1];
	workGroupMaxSize[2] = limits.maxComputeWorkGroupSize[2];

	maxInvocationSize = limits.maxComputeWorkGroupInvocations;
	invocationSize = invocationSizeIn;

	if(invocationSize > maxInvocationSize)
	{throw std::runtime_error("Invocation size too large!");}	

	// Create commande buffer and sync objects
	createDescriptorSetLayout(3);
	createCommandBuffer();
	createSyncObjects();
	createStagingBuffer();
}

Computer::~Computer()
{
	destroyStagingBuffer();
	destroySyncObjects();
	if(descriptorSetLayoutInit == true)
	{
		vkDestroyDescriptorPool(logicalDevice->getLogicalDevice(), inOutDescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice->getLogicalDevice(),descriptorSetLayout, nullptr);
		descriptorSetLayoutInit = false;
	}
	vkFreeCommandBuffers(logicalDevice->getLogicalDevice(), 
					logicalDevice->getCommandPool(), 1, &commandBuffer);
	vkFreeCommandBuffers(logicalDevice->getLogicalDevice(), 
					logicalDevice->getCommandPool(), 1, &memcpyCmdBuffer);
}

void Computer::createStagingBuffer()
{
	// Allocate staging buffer
	createBuffer(stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
					VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, gpuStaging, gpuMemory);
	vkMapMemory(logicalDevice->getLogicalDevice(), gpuMemory, 0, stagingSize, 0, &cpuStaging);
}

void Computer::destroyStagingBuffer()
{
	vkUnmapMemory(logicalDevice->getLogicalDevice(), gpuMemory);
	vkDestroyBuffer(logicalDevice->getLogicalDevice(),gpuStaging, nullptr);
	vkFreeMemory(logicalDevice->getLogicalDevice(), gpuMemory, nullptr);	
}

void Computer::createCommandBuffer()
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = logicalDevice->getCommandPool();
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkResult r;
	r = vkAllocateCommandBuffers(logicalDevice->getLogicalDevice(), &allocInfo, &commandBuffer);
	if (r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffer!");
	}

	r = vkAllocateCommandBuffers(logicalDevice->getLogicalDevice(), &allocInfo, &memcpyCmdBuffer);
	if (r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate memcpyCmdBuffer!");
	}

}

void Computer::createSyncObjects()
{
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	VkResult r;
	r = vkCreateFence(logicalDevice->getLogicalDevice(), 
					&fenceInfo, nullptr, &computeFence);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create fence!");
	}
	r = vkCreateFence(logicalDevice->getLogicalDevice(), 
					&fenceInfo, nullptr, &transferFence);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create fence!");
	}

	// Semaphores


    semaphoreTypeInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphoreTypeInfo.initialValue  = 0;

    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &semaphoreTypeInfo;
    
	vkCreateSemaphore(logicalDevice->getLogicalDevice(), &semaphoreCreateInfo, 
					nullptr, &computeSemaphore);
    vkCreateSemaphore(logicalDevice->getLogicalDevice(), &semaphoreCreateInfo, 
					nullptr, &transferSemaphore);
}

void Computer::destroySyncObjects()
{
    vkDestroyFence(logicalDevice->getLogicalDevice(), computeFence, nullptr);
    vkDestroyFence(logicalDevice->getLogicalDevice(), transferFence, nullptr);

	vkDestroySemaphore(logicalDevice->getLogicalDevice(), computeSemaphore, nullptr);
	vkDestroySemaphore(logicalDevice->getLogicalDevice(), transferSemaphore, nullptr);
}


//////////////////////////////////////////////////////////////////////////////////
//            ____                       _____                                  //
//           |  _ \ _ __ __ ___      __ |  ___| __ __ _ _ __ ___   ___          //
//           | | | | '__/ _` \ \ /\ / / | |_ | '__/ _` | '_ ` _ \ / _ \         //
//           | |_| | | | (_| |\ V  V /  |  _|| | | (_| | | | | | |  __/         //
//           |____/|_|  \__,_| \_/\_/   |_|  |_|  \__,_|_| |_| |_|\___|         //
//////////////////////////////////////////////////////////////////////////////////

//void Computer::recordCommandBuffer(VkCommandBuffer buffer, uint32_t dataLength)
void Computer::recordCommandBuffer(vkTools::ComputePipeline* pipeline, 
				VkCommandBuffer buffer, uint32_t dataLength)
{
	
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // Optional
	beginInfo.pInheritanceInfo = nullptr; // Optional

	uint32_t workGroupCount = (dataLength/invocationSize)+1;
	if(workGroupCount > workGroupMaxCount[0])
	{throw std::runtime_error("Too much data, workgroup count exceeds max!");}

	VkResult r;
	r = vkBeginCommandBuffer(buffer, &beginInfo);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to begin recording command buffer!");
	}

	vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());
	vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
					pipeline->getLayout(), 0, 1, 
					&descriptorSet, 0, nullptr);

	vkCmdPushConstants(buffer, pipeline->getLayout(), 
					VK_SHADER_STAGE_COMPUTE_BIT, 0, 
					sizeof(uint32_t), &dataLength);
	vkCmdDispatch(buffer, (dataLength/invocationSize)+1, 1, 1);	

	r = vkEndCommandBuffer(buffer);
	if (r != VK_SUCCESS) 
	{
		throw std::runtime_error("failed to record command buffer!");
	}
}

VkCommandBuffer Computer::getCommandBuffer(){return commandBuffer;}

void Computer::compute()
{
	//vkWaitForFences(logicalDevice->getLogicalDevice(), 1, &computeFence, VK_TRUE, UINT64_MAX);
	vkResetFences(logicalDevice->getLogicalDevice(), 1, &computeFence);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	
	submitInfo.signalSemaphoreCount = 0;
	
	VkBool32 r;
	r = vkQueueSubmit(logicalDevice->getComputeQueue(), 1, &submitInfo, 
					computeFence);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit compute command buffer!");
	}

	vkWaitForFences(logicalDevice->getLogicalDevice(), 1, &computeFence, VK_TRUE, UINT64_MAX);
}



//////////////////////////////////////////////////////////////////////////////////
//             ____         __  __             _____           _                //
//            | __ ) _   _ / _|/ _| ___ _ __  |_   _|__   ___ | |___            //
//            |  _ \| | | | |_| |_ / _ \ '__|   | |/ _ \ / _ \| / __|           //
//            | |_) | |_| |  _|  _|  __/ |      | | (_) | (_) | \__ \           //
//            |____/ \__,_|_| |_|  \___|_|      |_|\___/ \___/|_|___/           //
//////////////////////////////////////////////////////////////////////////////////
void Computer::vkMemcpy(void* dst, void* src, uint64_t size, uint64_t dstOffset, 
				uint64_t srcOffset, vkMemcpyFlags flag)
{
	if(size == 0){return;}
	VkQueue queue = logicalDevice->getTransferQueue();
	VkDevice logicalDev = logicalDevice->getLogicalDevice();
	
	VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	VkBufferCopy copyRegion{};
	//copyRegion.size = size;
	
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			
	VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

	uint32_t chunks = size/chunkSize;
	uint32_t remaining = size-(chunks*chunkSize);

	uint64_t semaphoreValue =  0;
	vkGetSemaphoreCounterValue(logicalDev, transferSemaphore, &semaphoreValue);
	uint64_t waitValue =  semaphoreValue+1;
	
	VkTimelineSemaphoreSubmitInfo tssi{};
    tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;


    // Wire up wait
    tssi.waitSemaphoreValueCount   = (waitValue > 0) ? 1 : 0;
    tssi.pWaitSemaphoreValues      = (waitValue > 0) ? &waitValue   : nullptr;

    // Wire up signal
    tssi.signalSemaphoreValueCount = 1;
    tssi.pSignalSemaphoreValues    = (signalValue > 0) ? &signalValue : nullptr;

    VkSubmitInfo si{};
    si.sType  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext  = &tssi;

    si.waitSemaphoreCount   = (waitValue   > 0) ? 1 : 0;
    si.pWaitSemaphores      = (waitValue   > 0) ? &sem : nullptr;
    si.pWaitDstStageMask    = (waitValue   > 0) ? &waitStage : nullptr;

    si.signalSemaphoreCount = (signalValue > 0) ? 1 : 0;
    si.pSignalSemaphores    = (signalValue > 0) ? &sem : nullptr;

    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
	
	switch(flag)
	{
		case HostToDevice:
			for(uint32_t i=0;i<chunks;i++)
			{	
				// Map memory and transfer from src to staging
				memcpy(cpuStaging, (void*) ((uint8_t*) src+srcOffset+i*chunkSize), chunkSize);
			
				// Record command buffer 
				copyRegion.srcOffset = 0;
				copyRegion.dstOffset = dstOffset+i*chunkSize;
				copyRegion.size = chunkSize;
				vkBeginCommandBuffer(memcpyCmdBuffer, &beginInfo);
				vkCmdCopyBuffer(memcpyCmdBuffer, gpuStaging, (VkBuffer) dst, 1, &copyRegion);
				vkEndCommandBuffer(memcpyCmdBuffer);

				// Submit to queue 
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &memcpyCmdBuffer;
				submitInfo.signalSemaphoreCount = 1;
        		submitInfo.pSignalSemaphores = &transferSemaphore;

				vkResetFences(logicalDev, 1, &transferFence);
				vkQueueSubmit(queue, 1, &submitInfo, transferFence);
				vkWaitForFences(logicalDev,1,&transferFence,VK_TRUE,UINT64_MAX);
			}
			if(remaining != 0)
			{
				// Map memory and transfer from src to staging
				memcpy(cpuStaging, (void*) ((uint8_t*) src+srcOffset+chunks*chunkSize), remaining);
			
				// Record command buffer 
				copyRegion.srcOffset = 0;
				copyRegion.dstOffset = dstOffset+chunks*chunkSize;
				copyRegion.size = remaining;
				vkBeginCommandBuffer(memcpyCmdBuffer, &beginInfo);
				vkCmdCopyBuffer(memcpyCmdBuffer, gpuStaging, (VkBuffer) dst, 1, &copyRegion);
				vkEndCommandBuffer(memcpyCmdBuffer);

				// Submit to queue 
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &memcpyCmdBuffer;
				vkResetFences(logicalDev, 1, &transferFence);
				vkQueueSubmit(queue, 1, &submitInfo, transferFence);
				vkWaitForFences(logicalDev,1,&transferFence,VK_TRUE,UINT64_MAX);
			}
			break;
	
		case DeviceToHost:	
			for(uint32_t i=0;i<chunks;i++)
			{	
				// Record command buffer 
				copyRegion.srcOffset = srcOffset+i*chunkSize;
				copyRegion.dstOffset = 0;
				copyRegion.size = chunkSize;
				vkBeginCommandBuffer(memcpyCmdBuffer, &beginInfo);
				vkCmdCopyBuffer(memcpyCmdBuffer, (VkBuffer) src, gpuStaging, 1, &copyRegion);
				vkEndCommandBuffer(memcpyCmdBuffer);

				// Submit to queue 
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &memcpyCmdBuffer;
				vkResetFences(logicalDev, 1, &transferFence);
				vkQueueSubmit(queue, 1, &submitInfo, transferFence);
				vkWaitForFences(logicalDev,1,&transferFence,VK_TRUE,UINT64_MAX);
			
				// Map memory and transfer from src to staging
				memcpy((void*) ((uint8_t*) dst+dstOffset+i*chunkSize), cpuStaging, chunkSize);
			}
			if(remaining != 0)
			{
				// Record command buffer 
				copyRegion.srcOffset = srcOffset+chunks*chunkSize;
				copyRegion.dstOffset = 0;
				copyRegion.size = remaining;
				vkBeginCommandBuffer(memcpyCmdBuffer, &beginInfo);
				vkCmdCopyBuffer(memcpyCmdBuffer, (VkBuffer) src, gpuStaging, 1, &copyRegion);
				vkEndCommandBuffer(memcpyCmdBuffer);

				// Submit to queue 
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &memcpyCmdBuffer;
				vkResetFences(logicalDev, 1, &transferFence);
				vkQueueSubmit(queue, 1, &submitInfo, transferFence);
				vkWaitForFences(logicalDev,1,&transferFence,VK_TRUE,UINT64_MAX);
			
				// Map memory and transfer from src to staging
				memcpy((void*) ((uint8_t*) dst+dstOffset+chunks*chunkSize), cpuStaging, remaining);
			}
			break;
	};
}


void Computer::createBuffer
(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult r;
	r = vkCreateBuffer(logicalDevice->getLogicalDevice(), &bufferInfo, nullptr, &buffer);
	
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create buffer!");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(logicalDevice->getLogicalDevice(), buffer, &memRequirements);

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(logicalDevice->getPhysicalDevice(), &memProperties);

	uint32_t typeFilter = memRequirements.memoryTypeBits;
	bool memoryCompatible = false;
	uint32_t memoryTypeIndex = 0;

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			memoryCompatible = true;
			memoryTypeIndex = i;
			break;
		}
	}

	if(memoryCompatible == false)
	{
		throw std::runtime_error("failed to find suitable memory!");
	}

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = memoryTypeIndex;

	r = vkAllocateMemory(logicalDevice->getLogicalDevice(), &allocInfo, nullptr, &bufferMemory);

	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate buffer memory!");
	}

	vkBindBufferMemory(logicalDevice->getLogicalDevice(), buffer, bufferMemory, 0);

}

void Computer::copyBuffer
(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, uint32_t dstOffset, uint32_t srcOffset, VkQueue queue)
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = logicalDevice->getCommandPool();
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(logicalDevice->getLogicalDevice(), &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = srcOffset; // Optional
	copyRegion.dstOffset = dstOffset; // Optional
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(logicalDevice->getLogicalDevice(), 
					logicalDevice->getCommandPool(), 1, &commandBuffer);
}

void Computer::fillBaseWriteDescriptorSet(uint32_t n, VkWriteDescriptorSet* writeDescriptorSet)
{
	// SSBO
	for(uint32_t i=0; i<n;i++)
	{
		writeDescriptorSet[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet[i].dstSet = descriptorSet;
		writeDescriptorSet[i].dstBinding = i;
		writeDescriptorSet[i].dstArrayElement = 0;
		writeDescriptorSet[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSet[i].descriptorCount = 1;
		//writeDescriptorSet[i].pBufferInfo = &bufferInfoInput1;
		writeDescriptorSet[i].pImageInfo = nullptr;
		writeDescriptorSet[i].pTexelBufferView = nullptr;
		writeDescriptorSet[i].pNext = nullptr;
	}
}

void Computer::createDescriptorSetLayout(uint32_t N)
{	
	VkDescriptorSetLayoutBinding* bindings;
	bindings = (VkDescriptorSetLayoutBinding*) malloc(N*sizeof(VkDescriptorSetLayoutBinding));
	
	for(uint32_t i=0;i<N;i++)
	{
    	bindings[i].binding = i;
    	bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    	bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings[i].pImmutableSamplers = nullptr;
	}	

	if(descriptorSetLayoutInit == true)
	{
		vkDestroyDescriptorPool(logicalDevice->getLogicalDevice(), inOutDescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice->getLogicalDevice(),descriptorSetLayout, nullptr);
		descriptorSetLayoutInit = false;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = N;
	layoutInfo.pBindings = bindings;

	VkResult r;
	r = vkCreateDescriptorSetLayout(logicalDevice->getLogicalDevice(), 
					&layoutInfo, nullptr, &descriptorSetLayout);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout!");
	}
	descriptorSetLayoutInit = true;

	VkDescriptorPoolSize poolSize;
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = N;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;

	r = vkCreateDescriptorPool(logicalDevice->getLogicalDevice(), 
					&poolInfo, nullptr, &inOutDescriptorPool); 
	
	if(r != VK_SUCCESS) 
	{
		throw std::runtime_error("failed to create descriptor pool!");
	}
	free(bindings);
	
	//pipeline->setLayoutDescriptors(1,&descriptorSetLayout);
	//pipeline->setPushConstants(VK_SHADER_STAGE_COMPUTE_BIT,sizeof(uint32_t),0);
	//pipeline->recreatePipeline();

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = inOutDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &descriptorSetLayout;

	r=vkAllocateDescriptorSets(logicalDevice->getLogicalDevice(),&allocInfo,&descriptorSet);
	if(r != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets!");
	}
}

VkDescriptorSetLayout* Computer::getDescriptorSetLayout(){return &descriptorSetLayout;}
