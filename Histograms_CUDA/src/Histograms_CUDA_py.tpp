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

