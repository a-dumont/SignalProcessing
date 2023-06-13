#include <future>
#include <stdexcept>

////////////////////////////////////////////////////////////////
//  _     _   _   _ _     _                                   //
// / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//                                 |___/                      //
////////////////////////////////////////////////////////////////
		
template<class DataType>
np_uint32 digitizer_histogram_1d_py(py::array_t<DataType,py::array::c_style> py_x)
{
	py::buffer_info buf_x = py_x.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_x.size;
	uint64_t hist_size = (uint64_t) 1<<(8*sizeof(DataType));

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<23)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+hist_size);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_1d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_hist,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,streams[1]);
		digitizer_histogram_1d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_hist,streams[1]);
	}
	else
	{
		digitizer_histogram_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{hist_size},
		{sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//    _     _   _   _ _     _                                    //
//   / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___    //
//   | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \   //
//   | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |  //
//   |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|  //
//	                                 |___/                       //
//                         _     _           _                   //
//               ___ _   _| |__ | |__  _   _| |_ ___             //
//              / __| | | | '_ \| '_ \| | | | __/ _ \            //
//              \__ \ |_| | |_) | |_) | |_| | ||  __/            //
//              |___/\__,_|_.__/|_.__/ \__, |\__\___|            //
//                                     |___/                     //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_subbyte_1d_py
(py::array_t<DataType,py::array::c_style> py_x, uint8_t nbits)
{
	py::buffer_info buf_x = py_x.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	if (nbits > 7 || nbits < 1)
	{
		throw std::runtime_error("U dumbdumb nbits must be between 1 and 7 inclusively.");
	}

	uint64_t N = buf_x.size;
	uint64_t hist_size = (uint64_t) 1<<nbits;

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<23)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+hist_size);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_subbyte_1d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_subbyte_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
		digitizer_histogram_subbyte_1d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_hist,nbits,streams[1]);
	}
	else
	{
		digitizer_histogram_subbyte_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{hist_size},
		{sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//    _     _   _   _ _     _                                    //
//   / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___    //
//   | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \   //
//   | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |  //
//   |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|  //
//	                                 |___/                       //
//                             _                                 //
//	                       ___| |_ ___ _ __                      //
//	                      / __| __/ _ \ '_ \                     //
//	                      \__ \ ||  __/ |_) |                    //
//	                      |___/\__\___| .__/                     //
//	                                  |_|                        //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_step_1d_py
(py::array_t<DataType,py::array::c_style> py_x, uint8_t nbits)
{
	py::buffer_info buf_x = py_x.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	if (nbits > 7 || nbits < 1)
	{
		throw std::runtime_error("U dumbdumb nbits must be between 1 and 7 inclusively.");
	}

	uint64_t N = buf_x.size;
	uint64_t hist_size = (uint64_t) 1<<nbits;
	uint64_t total_size = hist_size+(hist_size<<nbits);

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<23)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(total_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, total_size*sizeof(uint32_t)+N*sizeof(DataType));
	cudaMemset(gpu_hist,0,total_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+total_size);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_step_1d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_step_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
		digitizer_histogram_step_1d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_hist,nbits,streams[1]);
	}
	else
	{
		digitizer_histogram_step_1d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,nbits,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,total_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{hist_size+1,hist_size},
		{hist_size*sizeof(uint32_t),sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//    _     _   _   _ _     _                                    //
//   / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___    //
//   | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \   //
//   | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |  //
//   |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|  //
//	                                 |___/                       //
//                   _  ___    _     _ _                         //
//                  / |/ _ \  | |__ (_) |_ ___                   //
//                  | | | | | | '_ \| | __/ __|                  //
//                  | | |_| | | |_) | | |_\__ \                  //
//                  |_|\___/  |_.__/|_|\__|___/                  //
/////////////////////////////////////////////////////////////////// 

template<class DataType>
np_uint32 digitizer_histogram_10bits_1d_py(py::array_t<DataType,py::array::c_style> py_x)
{
	py::buffer_info buf_x = py_x.request();

	if (buf_x.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = buf_x.size;
	uint64_t hist_size = (uint64_t) 1<<10;

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<23)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	uint16_t* ptr_x = (uint16_t*) buf_x.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	uint16_t* gpu_x = (uint16_t*) (gpu_hist+hist_size);

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_1d<uint16_t>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_hist,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_1d<uint16_t>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,streams[1]);
		digitizer_histogram_1d<uint16_t>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_hist,streams[1]);
	}
	else
	{
		digitizer_histogram_1d<uint16_t>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_hist,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{hist_size},
		{sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_2d_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t axis_size = (uint64_t) 1<<(8*sizeof(DataType));
	uint64_t hist_size = (uint64_t) 1<<(16*sizeof(DataType));

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<25)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	DataType* ptr_y = (DataType*) buf_y.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+2*N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+hist_size);
	DataType* gpu_y = gpu_x+N;

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	cudaMemcpyAsync(gpu_y,ptr_y,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+i*transfers_N,ptr_y+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_2d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_y+(i-1)*transfers_N,gpu_hist,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+transfers*transfers_N,ptr_y+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,streams[1]);
		digitizer_histogram_2d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_y+transfers*transfers_N,gpu_hist,streams[1]);
	}
	else
	{
		digitizer_histogram_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{axis_size,axis_size},
		{sizeof(uint32_t)*axis_size,sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//                         _     _           _                   //
//               ___ _   _| |__ | |__  _   _| |_ ___             //
//              / __| | | | '_ \| '_ \| | | | __/ _ \            //
//              \__ \ |_| | |_) | |_) | |_| | ||  __/            //
//              |___/\__,_|_.__/|_.__/ \__, |\__\___|            //
//                                     |___/                     //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_subbyte_2d_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y, uint8_t nbits)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	if (nbits > 7 || nbits < 1)
	{
		throw std::runtime_error("U dumbdumb nbits must be between 1 and 7 inclusively.");
	}

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t axis_size = (uint64_t) 1<<nbits;
	uint64_t hist_size = axis_size<<nbits;

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<25)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	DataType* ptr_y = (DataType*) buf_y.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+2*N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+hist_size);
	DataType* gpu_y = gpu_x+N;

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	cudaMemcpyAsync(gpu_y,ptr_y,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+i*transfers_N,ptr_y+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_subbyte_2d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_y+(i-1)*transfers_N,gpu_hist,nbits,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+transfers*transfers_N,ptr_y+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_subbyte_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,nbits,streams[1]);
		digitizer_histogram_subbyte_2d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_y+transfers*transfers_N,gpu_hist,nbits,streams[1]);
	}
	else
	{
		digitizer_histogram_subbyte_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,nbits,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{axis_size,axis_size},
		{sizeof(uint32_t)*axis_size,sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//                             _                                 //
//	                       ___| |_ ___ _ __                      //
//	                      / __| __/ _ \ '_ \                     //
//	                      \__ \ ||  __/ |_) |                    //
//	                      |___/\__\___| .__/                     //
//	                                  |_|                        //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_step_2d_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y, uint8_t nbits)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}
	if (nbits > 7 || nbits < 1)
	{
		throw std::runtime_error("U dumbdumb nbits must be between 1 and 7 inclusively.");
	}

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t axis_size = (uint64_t) 1<<nbits;
	uint64_t hist_size = axis_size<<nbits;
	uint64_t total_size = hist_size+(hist_size<<nbits<<nbits);

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<25)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	DataType* ptr_x = (DataType*) buf_x.ptr;
	DataType* ptr_y = (DataType*) buf_y.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(total_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, total_size*sizeof(uint32_t)+2*N*sizeof(DataType));
	cudaMemset(gpu_hist,0,total_size*sizeof(uint32_t));
	
	DataType* gpu_x = (DataType*) (gpu_hist+total_size);
	DataType* gpu_y = gpu_x+N;

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	cudaMemcpyAsync(gpu_y,ptr_y,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+i*transfers_N,ptr_y+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_step_2d<DataType>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_y+(i-1)*transfers_N,gpu_hist,nbits,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+transfers*transfers_N,ptr_y+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_step_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,nbits,streams[1]);
		digitizer_histogram_step_2d<DataType>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_y+transfers*transfers_N,gpu_hist,nbits,streams[1]);
	}
	else
	{
		digitizer_histogram_step_2d<DataType>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,nbits,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,total_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{hist_size+1,axis_size,axis_size},
		{sizeof(uint32_t)*hist_size,sizeof(uint32_t)*axis_size,sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//                   _  ___    _     _ _                         //
//                  / |/ _ \  | |__ (_) |_ ___                   //
//                  | | | | | | '_ \| | __/ __|                  //
//                  | | |_| | | |_) | | |_\__ \                  //
//                  |_|\___/  |_.__/|_|\__|___/                  //
///////////////////////////////////////////////////////////////////

template<class DataType>
np_uint32 digitizer_histogram_10bits_2d_py(py::array_t<DataType,py::array::c_style> py_x,
				py::array_t<DataType,py::array::c_style> py_y)
{
	py::buffer_info buf_x = py_x.request();
	py::buffer_info buf_y = py_y.request();

	if (buf_x.ndim != 1 || buf_y.ndim != 1)
	{
		throw std::runtime_error("U dumbdumb dimension must be 1.");
	}

	uint64_t N = std::min(buf_x.size,buf_y.size);
	uint64_t axis_size = (uint64_t) 1<<10;
	uint64_t hist_size = (uint64_t) 1<<20;

	uint64_t data_size = N*sizeof(DataType);
	uint64_t transfer_size = std::min(((uint64_t)(1<<25)),data_size);
	uint64_t transfers = data_size/transfer_size;
	uint64_t transfers_N = transfer_size/sizeof(DataType);
	uint64_t remaining = data_size-transfers*transfer_size;
	uint64_t remaining_N = remaining/sizeof(DataType);

	uint16_t* ptr_x = (uint16_t*) buf_x.ptr;
	uint16_t* ptr_y = (uint16_t*) buf_y.ptr;
	
	uint32_t* hist = (uint32_t*) malloc(hist_size*sizeof(uint32_t));

	uint32_t* gpu_hist;
	cudaMalloc((void**)&gpu_hist, hist_size*sizeof(uint32_t)+2*N*sizeof(DataType));
	cudaMemset(gpu_hist,0,hist_size*sizeof(uint32_t));
	
	uint16_t* gpu_x = (uint16_t*) (gpu_hist+hist_size);
	uint16_t* gpu_y = gpu_x+N;

	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	

	cudaMemcpyAsync(gpu_x,ptr_x,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	cudaMemcpyAsync(gpu_y,ptr_y,transfer_size,cudaMemcpyHostToDevice,streams[0]);
	for(uint64_t i=1;i<transfers;i++)
	{
		cudaMemcpyAsync(gpu_x+i*transfers_N,ptr_x+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+i*transfers_N,ptr_y+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_10bits_2d<uint16_t>(transfers_N,gpu_x+(i-1)*transfers_N,
						gpu_y+(i-1)*transfers_N,gpu_hist,streams[1]);
	}
	if(remaining!=0)
	{
		cudaMemcpyAsync(gpu_x+transfers*transfers_N,ptr_x+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		cudaMemcpyAsync(gpu_y+transfers*transfers_N,ptr_y+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
		digitizer_histogram_10bits_2d<uint16_t>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,streams[1]);
		digitizer_histogram_10bits_2d<uint16_t>(remaining_N,gpu_x+transfers*transfers_N, 
						gpu_y+transfers*transfers_N,gpu_hist,streams[1]);
	}
	else
	{
		digitizer_histogram_10bits_2d<uint16_t>(transfers_N,gpu_x+(transfers-1)*transfers_N,
						gpu_y+(transfers-1)*transfers_N,gpu_hist,streams[1]);
	}
	cudaMemcpy(hist,gpu_hist,hist_size*sizeof(uint32_t),cudaMemcpyDeviceToHost);	

	cudaFree(gpu_hist);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

	py::capsule free_when_done( hist, free );
	return np_uint32 
	(
		{axis_size,axis_size},
		{sizeof(uint32_t)*axis_size,sizeof(uint32_t)},
		hist,
		free_when_done	
	);
}

///////////////////////////////////////
//       _                           //
//   ___| | __ _ ___ ___  ___  ___   //
//  / __| |/ _` / __/ __|/ _ \/ __|  //
// | (__| | (_| \__ \__ \  __/\__ \  //
//  \___|_|\__,_|___/___/\___||___/  //
//                                   //
///////////////////////////////////////                                      


class Digitizer_histogram2D_step_CUDA_py
{
	protected:
		uint32_t *hist_cpu, *hist_gpu, *hist_merged;
		uint8_t *x_gpu, *y_gpu;
		uint64_t nbits, size, total_size, count;
		uint64_t N_t, N, N_cpu, N_gpu, N_p, data_size;
		uint64_t transfers,transfer_size,transfers_N,remaining,remaining_N;
		cudaStream_t streams[2];

		void digitizer_histogram2D_step(uint32_t* hist, uint8_t* data_x, 
								uint8_t* data_y, uint64_t N, uint8_t nbits)
		{
			uint8_t shift = 8-nbits;
   			uint64_t n2 = 2*nbits;
   			uint64_t n3 = 3*nbits;
	
			uint32_t* hist2 = hist+(1<<n2);

			uint64_t bin_x, bin_y, bin_x2, bin_y2;

			for(uint64_t i=0; i<(N-1); i++)
			{
				bin_x = data_x[i] >> shift;
				bin_y = data_y[i] >> shift;
				bin_x2 = data_x[i+1] >> shift;
				bin_y2 = data_y[i+1] >> shift;
		
				hist[bin_y ^ (bin_x<<nbits)] += 1;
				hist2[bin_y2 ^ (bin_x<<n3) ^ (bin_y<<n2) ^ (bin_x2<<nbits)] += 1;
			}
		}

		void accumulate_cpu(uint8_t* xdata, uint8_t* ydata)
		{
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t i=0;i<N_t;i++)
			{
				manage_thread_affinity();
				digitizer_histogram2D_step(hist_cpu+i*total_size,xdata+i*N_p,ydata+i*N_p,N_p,nbits);
			}
		}
		
		void accumulate_gpu(uint8_t* xdata, uint8_t* ydata)
		{
			cudaMemcpyAsync(x_gpu,xdata,transfer_size,cudaMemcpyHostToDevice,streams[0]);
			cudaMemcpyAsync(y_gpu,ydata,transfer_size,cudaMemcpyHostToDevice,streams[0]);
			for(uint64_t i=1;i<transfers;i++)
			{
				cudaMemcpyAsync(x_gpu+i*transfers_N,xdata+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
				cudaMemcpyAsync(y_gpu+i*transfers_N,ydata+i*transfers_N,
						transfer_size,cudaMemcpyHostToDevice,streams[0]);
				digitizer_histogram_step_2d(transfers_N,x_gpu+(i-1)*transfers_N,
						y_gpu+(i-1)*transfers_N,hist_gpu,nbits,streams[1]);
			}
			if(remaining!=0)
			{
				cudaMemcpyAsync(x_gpu+transfers*transfers_N,xdata+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
				cudaMemcpyAsync(y_gpu+transfers*transfers_N,ydata+transfers*transfers_N,
						remaining,cudaMemcpyHostToDevice,streams[0]);
				digitizer_histogram_step_2d(transfers_N,x_gpu+(transfers-1)*transfers_N,
						y_gpu+(transfers-1)*transfers_N,hist_gpu,nbits,streams[1]);
				digitizer_histogram_step_2d(remaining_N,x_gpu+transfers*transfers_N, 
						y_gpu+transfers*transfers_N,hist_gpu,nbits,streams[1]);
			}
			else
			{
				digitizer_histogram_step_2d(transfers_N,x_gpu+(transfers-1)*transfers_N,
						y_gpu+(transfers-1)*transfers_N,hist_gpu,nbits,streams[1]);
			}
		}

		void manage_thread_affinity()
		{
			//Shamelessly stolen from Jean-Olivier's code at
			//https://github.com/JeanOlivier/Histograms-OTF/blob/master/histograms.c
			#ifdef _WIN32_WINNT
				int nbgroups = GetActiveProcessorGroupCount();
				int *threads_per_groups = (int *) malloc(nbgroups*sizeof(int));
	    		for (int i=0; i<nbgroups; i++)
	    		{
					threads_per_groups[i] = GetActiveProcessorCount(i);
				}
	    		// Fetching thread number and assigning it to cores
				int tid = omp_get_thread_num(); // Internal omp thread number (0 -- OMP_NUM_THREADS)
	    		HANDLE thandle = GetCurrentThread();
	    		bool result;
				// We change group for each thread
				short unsigned int set_group = tid%nbgroups; 
                                                                          		
				// Nb of threads in group for affinity mask.
    			int nbthreads = threads_per_groups[set_group]; 
				// nbcores amount of 1 in binary
	    		GROUP_AFFINITY group = {((uint64_t)1<<nbthreads)-1, set_group};
				// Actually setting the affinity
				result = SetThreadGroupAffinity(thandle, &group, NULL); 
				if(!result) std::fprintf(stderr, "Failed setting output for tid=%i\n", tid);
				free(threads_per_groups);
				else
			//We let openmp and the OS manage the threads themselves
			#endif
		}
	public:
		Digitizer_histogram2D_step_CUDA_py(uint64_t nbits_in, uint64_t N_in)
		{
			if(nbits_in > (uint64_t) 6)
			{throw std::runtime_error("U dumbdumb nbits to large for parallel reduction.");}
			
			nbits = nbits_in;
			size = 1<<nbits;
			total_size = size*size*(size*size+1);
			count = 0;

			N = N_in;
			N_cpu = 4*(N>>3);
			N_gpu = 4*(N>>3);
			
			if((N_cpu+N_gpu) != N)
			{throw std::runtime_error("U dumbdumb N_in must be power of 2.");}
				
		    #ifdef _WIN32_WINNT
		    	uint64_t nbgroups = GetActiveProcessorGroupCount();
		        N_t = std::min((uint64_t) 64,omp_get_max_threads()*nbgroups);
		    #else
		        N_t = omp_get_max_threads();
			#endif

			N_p = N/N_t;

			data_size = N_gpu*sizeof(uint8_t);

			transfer_size = std::min(((uint64_t)(1<<25)),data_size);
			transfers = data_size/transfer_size;
			transfers_N = transfer_size/sizeof(uint8_t);
			
			remaining = data_size-transfers*transfer_size;
			remaining_N = remaining/sizeof(uint8_t);
		            
			hist_cpu = (uint32_t*) malloc(N_t*sizeof(uint32_t)*total_size);
			std::memset(hist_cpu,0,N_t*sizeof(uint32_t)*total_size);
			
			hist_merged = (uint32_t*) malloc(sizeof(uint32_t)*total_size);
			std::memset(hist_merged,0,sizeof(uint32_t)*total_size);

			cudaMalloc((void**)&hist_gpu, total_size*sizeof(uint32_t));
			cudaMemset(hist_gpu,0,total_size*sizeof(uint32_t));
			
			cudaMalloc((void**)&x_gpu, N_gpu*sizeof(uint8_t));
			cudaMalloc((void**)&y_gpu, N_gpu*sizeof(uint8_t));

			cudaStreamCreate(&streams[0]);
			cudaStreamCreate(&streams[1]);
		}
		
		~Digitizer_histogram2D_step_CUDA_py()
		{
			free(hist_cpu);
			free(hist_merged);
			cudaFree(hist_gpu);
			cudaFree(x_gpu);
			cudaFree(y_gpu);
			cudaStreamDestroy(streams[0]);
			cudaStreamDestroy(streams[1]);
		}

		void accumulate(np_uint8 xdata_py, np_uint8 ydata_py)
		{
			uint8_t* xdata_cpu = (uint8_t*) xdata_py.request().ptr;	
			uint8_t* ydata_cpu = (uint8_t*) ydata_py.request().ptr;

			uint8_t* xdata_gpu = xdata_cpu+N_cpu;	
			uint8_t* ydata_gpu = ydata_cpu+N_cpu;	
			
			#pragma omp sections
			{
				#pragma omp section 
				{
					accumulate_cpu(xdata_cpu,ydata_cpu);
				}
				#pragma omp section
				{
					accumulate_gpu(xdata_gpu,ydata_gpu);	
				}
			}
			count += 1;
		}
		
		void resetHistogram()
		{
			std::memset(hist_cpu,0,N_t*sizeof(uint32_t)*total_size);
			std::memset(hist_merged,0,sizeof(uint32_t)*total_size);
			cudaMemset(hist_gpu,0,sizeof(uint32_t)*total_size);

			count = 0;
		}
		
		np_uint64 getHistogram()
		{
			uint64_t* hist_out;
			hist_out = (uint64_t*) malloc(sizeof(uint64_t)*total_size);
			std::memset(hist_out,0,sizeof(uint64_t)*total_size);

			cudaMemcpy2D(hist_out,
							sizeof(uint64_t),
							hist_gpu,
							sizeof(uint32_t),
							total_size,
						    1,	
							cudaMemcpyDeviceToHost);

			#pragma omp parallel for num_threads(N_t)
			for(uint64_t j=0;j<N_t;j++)
			{
				manage_thread_affinity();
				for(uint64_t i=0;i<total_size;i++)
				{
					#pragma omp atomic
					hist_out[i] += hist_cpu[j*total_size+i];
				}
			}


			py::capsule free_when_done( hist_out, free );
			return np_uint64 
			(
			{(size*size)+1,size,size},
			{sizeof(uint64_t)*size*size,sizeof(uint64_t)*size,sizeof(uint64_t)},
			hist_out,
			free_when_done	
			);
		}
		
		uint64_t getCount(){return count;}
		uint64_t getNbits(){return nbits;}
		uint64_t getSize(){return size;}
		uint64_t getThreads(){return N_t;}
};

