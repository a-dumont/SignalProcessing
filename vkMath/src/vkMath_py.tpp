#include "vkMath_py.h"

template<typename Datatype>
py::cpp_function vkAddBuilder(vkComputer::Computer* computer, ComputePipelinePy* pipeline)
{
	return py::cpp_function([=](py::array_t<Datatype,py::array::c_style> in1, 
					py::array_t<Datatype,py::array::c_style> in2)
	{return vkAdd<Datatype>(computer,pipeline,in1,in2);});
}

template<typename Datatype>
py::array_t<Datatype,py::array::c_style> 
vkAdd(vkComputer::Computer* computer, ComputePipelinePy* pipeline, 
		py::array_t<Datatype,py::array::c_style> in1, 
		py::array_t<Datatype,py::array::c_style> in2)
{
	py::buffer_info buf1 = in1.request();
	py::buffer_info buf2 = in2.request();

	if (buf1.ndim !=  buf2.ndim)
	{
		throw std::runtime_error("U dumbdumb inputs must have same dimensions.");
	}	
	for(uint32_t i=0;i<buf1.ndim;i++)
	{
		if (buf1.shape[i] !=  buf2.shape[i])
		{
			throw std::runtime_error("U dumbdumb inputs must have same shape.");
		}
	}
	
	// Get pointers from python
	Datatype* ptr1 = (Datatype*) buf1.ptr;	
	Datatype* ptr2 = (Datatype*) buf2.ptr;	

	VkDevice logicalDev = pipeline->getLogicalDevice()->getLogicalDevice();

	// Pipeline initialization
	pipeline->setLayoutDescriptors(1,computer->getDescriptorSetLayout());
	pipeline->setPushConstants(VK_SHADER_STAGE_COMPUTE_BIT,sizeof(Datatype),0);
	pipeline->recreatePipeline();

	// Memory allocation
	vkComputer::vkMemcpyFlags HostToDevice = vkComputer::HostToDevice;
	vkComputer::vkMemcpyFlags DeviceToHost = vkComputer::DeviceToHost;
	uint32_t dataSize = buf1.size;
	uint32_t chunkSize = (1<<26)/sizeof(Datatype);
	uint32_t chunks = dataSize/chunkSize;
	uint32_t remainingSize = dataSize-(chunks*chunkSize);
	if(chunks==0){chunkSize = dataSize; chunks = 1; remainingSize = 0;}

	py::array_t<Datatype,py::array::c_style> result(buf1.shape);
	Datatype* out = (Datatype*) result.request().ptr;

	VkBuffer gpuBuffer;
	VkDeviceMemory gpuMemory;
	Datatype* cpuMemory;

	// Create gpuBuffer using gpuMemory
	computer->createBuffer(2*sizeof(Datatype)*chunkSize,
					VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |  
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT |	
					VK_BUFFER_USAGE_TRANSFER_DST_BIT,	
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					//VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				    //VK_MEMORY_PROPERTY_HOST_CACHED_BIT	|
					//VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
					gpuBuffer, gpuMemory);
	
	// CPU-GPU memory mapping
	//vkMapMemory(logicalDev, gpuMemory, 0, 2*sizeof(Datatype)*chunkSize, 0, (void**)&cpuMemory);

	// Define buffer usage
	VkDescriptorBufferInfo bufferInfoInput[2]; // Single buffer split into In1, In2, Out
	VkWriteDescriptorSet descriptorWrite[2]; 
	computer->fillBaseWriteDescriptorSet(2,descriptorWrite);

	bufferInfoInput[0].buffer = gpuBuffer;
	bufferInfoInput[0].offset = 0;
	bufferInfoInput[0].range = sizeof(Datatype)*chunkSize;
	descriptorWrite[0].pBufferInfo = &bufferInfoInput[0];
	
	bufferInfoInput[1].buffer = gpuBuffer;
	bufferInfoInput[1].offset = sizeof(Datatype)*chunkSize;
	bufferInfoInput[1].range = sizeof(Datatype)*chunkSize;
	descriptorWrite[1].pBufferInfo = &bufferInfoInput[1];

	// Inform the gpu of the buffers
	vkUpdateDescriptorSets(logicalDev, 2, descriptorWrite, 0, nullptr);

	// Record command buffer	
	computer->recordCommandBuffer(pipeline,computer->getCommandBuffer(),chunkSize);

	// First chunk memory transfer from cpu to gpu using mapped memory
	//std::memcpy(cpuMemory,ptr1,chunkSize*sizeof(Datatype));	
	//std::memcpy(cpuMemory+chunkSize,ptr2,chunkSize*sizeof(Datatype));
	
	computer->vkMemcpy(gpuBuffer,ptr1,1<<26,0,0,HostToDevice);
	computer->vkMemcpy(gpuBuffer,ptr2,1<<26,0,1<<26,HostToDevice);

	// Process all chunks
	for(uint32_t i=1;i<chunks;i++)
	{
		// Compute chunk i-1 transfered previously
		computer->compute();
		
		// Copy buffer to working memory
		//std::memcpy(out+(i-1)*chunkSize,cpuMemory,chunkSize*sizeof(Datatype));
		computer->vkMemcpy(out,gpuBuffer,1<<26,0,(i-1)*(1<<26),DeviceToHost);
		
		// Copy next chunk
		//std::memcpy(cpuMemory+chunkSize,ptr2+i*chunkSize,chunkSize*sizeof(Datatype));	
		//std::memcpy(cpuMemory,ptr1+i*chunkSize,chunkSize*sizeof(Datatype));	
	
		computer->vkMemcpy(gpuBuffer,ptr1,1<<26,i*(1<<26),0,HostToDevice);
		computer->vkMemcpy(gpuBuffer,ptr2,1<<26,i*(1<<26),1<<26,HostToDevice);

	}
	computer->compute();
	//std::memcpy(out+(chunks-1)*chunkSize,cpuMemory,chunkSize*sizeof(Datatype));
	computer->vkMemcpy(out,gpuBuffer,1<<26,0,chunks*(1<<26),DeviceToHost);

	// Remaining data after chunks
	/*
	if(remainingSize != 0)
	{
		computer->recordCommandBuffer(pipeline,computer->getCommandBuffer(),remainingSize);
		std::memcpy(cpuMemory,ptr1+chunks*chunkSize,remainingSize*sizeof(Datatype));	
		std::memcpy(cpuMemory+chunkSize,ptr2+chunks*chunkSize,remainingSize*sizeof(Datatype));
		computer->compute();
		std::memcpy(out+chunks*chunkSize,cpuMemory,remainingSize*sizeof(Datatype));
	}
	*/

	// Cleanup
	//vkUnmapMemory(logicalDev,gpuMemory);
	vkDestroyBuffer(logicalDev, gpuBuffer, nullptr);
	vkFreeMemory(logicalDev, gpuMemory, nullptr);

	return result; 
}
