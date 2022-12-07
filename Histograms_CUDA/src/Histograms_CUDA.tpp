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
    #else
        //We let openmp and the OS manage the threads themselves
    #endif
}

template<class DataType>
void filter_single(int64_t N, DataType* data, float* filter, DataType offset)
{
	float* gpu;
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	
	cufftHandle plan;
	
	cudaMalloc((void**)&gpu, 3*(N/2+1)*sizeof(float));	
	cudaMemcpyAsync(gpu+(N+2),data,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert<DataType,float>(N,reinterpret_cast<DataType*>(gpu+N+2),gpu,1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(N+2),filter,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	if (cufftPlan1d(&plan, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan, reinterpret_cast<cufftReal*>(gpu), 
							reinterpret_cast<cufftComplex*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}

	filter_CUDA<float>(N/2+1,gpu,gpu+(N+2),streams[0]);

	if (cufftPlan1d(&plan, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(gpu), 
							reinterpret_cast<cufftReal*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	} 
	
	rconvert<DataType,float>(N,gpu,reinterpret_cast<DataType*>(gpu),1.0/N,offset,streams[0]);
	cudaMemcpyAsync(data,gpu,sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	cufftDestroy(plan);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);
	cudaFree(gpu);
}

template<class DataType>
void filter_dual(int64_t N, DataType* data1, DataType* data2, 
				float* filter1, float* filter2, DataType offset)
{
	float* gpu;
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);	
	cufftHandle plan1, plan2;
	cufftSetStream(plan1, streams[0]);	
	cufftSetStream(plan2, streams[1]);	
	
	cudaMalloc((void**)&gpu, 6*(N/2+1)*sizeof(float));	
	cudaMemcpyAsync(gpu,data1,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert(N,reinterpret_cast<DataType*>(gpu),gpu,1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(N+2),filter1,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	cudaMemcpyAsync(gpu+(3*N/2+3),data2,sizeof(DataType)*N,cudaMemcpyHostToDevice,streams[0]);
	convert(N,reinterpret_cast<DataType*>(gpu+(3*N/2+3)),gpu+(3*N/2+3),1.0,offset,streams[0]);
	cudaMemcpyAsync(gpu+(5*N/2+5),filter2,sizeof(float)*N/2+1,cudaMemcpyHostToDevice,streams[1]);

	if (cufftPlan1d(&plan1, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan1, reinterpret_cast<cufftReal*>(gpu), 
							reinterpret_cast<cufftComplex*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan1);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}
	if (cufftPlan1d(&plan2, N, CUFFT_R2C, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecR2C(plan2, reinterpret_cast<cufftReal*>(gpu+(3*N/2+3)), 
							reinterpret_cast<cufftComplex*>(gpu+(3*N/2+3))) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan2);
		throw std::runtime_error("CUFFT error: ExecR2C failed");
	}

	filter_CUDA<float>(N/2+1,gpu,filter1,streams[0]);
	filter_CUDA<float>(N/2+1,gpu+(3*N/2+3),filter2,streams[1]);

	if (cufftPlan1d(&plan1, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan1, reinterpret_cast<cufftComplex*>(gpu), 
							reinterpret_cast<cufftReal*>(gpu)) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan1);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	}	
	if (cufftPlan1d(&plan2, N, CUFFT_C2R, 1) != CUFFT_SUCCESS)
	{
		throw std::runtime_error("CUFFT error: Plan creation failed");
	}
	if (cufftExecC2R(plan2, reinterpret_cast<cufftComplex*>(gpu+(3*N/2+3)), 
							reinterpret_cast<cufftReal*>(gpu+(3*N/2+3))) != CUFFT_SUCCESS)
	{
		cufftDestroy(plan2);
		throw std::runtime_error("CUFFT error: ExecC2R failed");
	} 

	
	rconvert(N,gpu,reinterpret_cast<DataType*>(gpu),1.0/N,offset,streams[0]);
	cudaMemcpyAsync(data1,gpu,sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	rconvert(N,gpu+(3*N/2+3),reinterpret_cast<DataType*>(gpu+(3*N/2+3)),1.0/N,offset,streams[1]);
	cudaMemcpyAsync(data2,gpu+(3*N/2+3),sizeof(DataType)*N,cudaMemcpyDeviceToHost,streams[0]);

	cufftDestroy(plan1);
	cufftDestroy(plan2);
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);
	cudaFree(gpu);
}


template<class DataType>
void digitizer_histogram(uint32_t* hist, DataType* data, uint64_t N)
{
	uint64_t size = 1<<(sizeof(DataType)*8);
	#pragma omp parallel for reduction(+:hist[:size])
	for(uint64_t i=0; i<N; i++)
	{
		hist[data[i]] += 1;
	}
}

template<class DataType>
void digitizer_histogram_subbyte(uint32_t* hist, DataType* data, uint64_t N, int nbits)
{
	uint8_t shift = sizeof(DataType)*8-nbits;
	#pragma omp parallel for reduction(+:hist[:1<<nbits])
	for(uint64_t i=0; i<N; i++)
	{
		hist[data[i] >> shift] += 1;
	}
}

template<class DataType>
void digitizer_histogram2D(uint32_t* hist, DataType* data_x, DataType* data_y, uint64_t N)
{
	uint64_t size = 1<<(8*sizeof(DataType));
	#pragma omp parallel for reduction(+:hist[:size*size])
	for(uint64_t i=0; i<N; i++)
	{
		hist[data_y[i]+size*data_x[i]] += 1;
	}
}

template<class DataType>
void digitizer_histogram2D_subbyte(uint32_t* hist, DataType* data_x, 
				DataType* data_y, uint64_t N, uint64_t nbits)
{
	uint64_t size = 1<<nbits;
	uint8_t shift = sizeof(DataType)*8-nbits;
	#pragma omp parallel for reduction(+:hist[:size<<nbits])
	for(uint64_t i=0; i<N; i++)
	{
		hist[(data_y[i]>>shift)+size*(data_x[i]>>shift)] += 1;
	}
}

template<class DataType>
void digitizer_histogram2D_steps(uint32_t* hist, DataType* data_x, 
				DataType* data_y, uint64_t N, uint8_t nbits, uint8_t steps)
{
	uint8_t shift = sizeof(DataType)*8-nbits;
	uint64_t s = 1<<nbits;
   	uint64_t s2 = s<<nbits;
   	uint64_t s3 = s2<<nbits;
   	uint64_t s4 = s3<<nbits;
	uint64_t s5 = steps*s4+s2;
		
	uint64_t bin_x, bin_y, bin_x2, bin_y2, bin_x3, bin_y3;

	for(uint64_t i=steps; i<(N-steps); i++)
	{
		bin_x = data_x[i] >> shift;
		bin_y = data_y[i] >> shift;
		hist[bin_y+s*bin_x] += 1;
		for(uint64_t j=1; j<(uint8_t)(steps+1);j++)
		{
			bin_x2 = data_x[i+j] >> shift;
			bin_y2 = data_y[i+j] >> shift;
			bin_x3 = data_x[i-j] >> shift;
			bin_y3 = data_y[i-j] >> shift;

			hist[s2+bin_x*s3+bin_y*s2+s*bin_x2+bin_y2+(j-1)*s4]+=1;
			hist[s5+bin_x*s3+bin_y*s2+s*bin_x3+bin_y3+(j-1)*s4]+=1;
		}
	}
}

class cdigitizer_histogram2D_steps
{
	protected:
		uint32_t* hist;
		uint64_t* hist_out;
		bool hist_out_init;
		uint64_t nbits;
		uint64_t size;
		uint64_t steps;
		uint64_t count;
		uint64_t N_t;
	public:
		cdigitizer_histogram2D_steps(uint64_t nbits_in, uint64_t steps_in)
		{
			if(nbits_in > (uint64_t) 10)
			{throw std::runtime_error("U dumbdumb nbits to large for parallel reduction.");}
			nbits = nbits_in;
			size = 1<<nbits;
			steps = steps_in;
			count = 0;
            #ifdef _WIN32_WINNT
                uint64_t nbgroups = GetActiveProcessorGroupCount();
                N_t = std::min((uint64_t) 64,omp_get_max_threads()*nbgroups);
            #else
                N_t = omp_get_max_threads();
			#endif
            
			hist = (uint32_t*) malloc(N_t*sizeof(uint32_t)*size*size*(2*steps*size*size+1));

			std::memset(hist,0,N_t*sizeof(uint32_t)*size*size*(2*steps*size*size+1));
			hist_out_init = false;
		}
		
		~cdigitizer_histogram2D_steps(){free(hist);if(hist_out_init == true){free(hist_out);}}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			uint64_t total_size = size*size*(2*steps*size*size+1);
			N = N/N_t;
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t i=0;i<N_t;i++)
			{
                manage_thread_affinity();
				digitizer_histogram2D_steps(hist+i*total_size,xdata+i*N,ydata+i*N,N,nbits,steps);
			}
			count += 1;
		}
		void resetHistogram()
		{
			std::memset(hist,0,N_t*sizeof(uint32_t)*size*size*(2*steps*size*size+1));
			if(hist_out_init == true)
			{
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(2*steps*size*size+1));
			}
			count = 0;
		}
		uint64_t* getHistogram()
		{
			if(hist_out_init == false)
			{
				hist_out = (uint64_t*) malloc(sizeof(uint32_t)*size*size*(2*steps*size*size+1));
				std::memset(hist_out,0,sizeof(uint32_t)*size*size*(2*steps*size*size+1));
				hist_out_init = true;
			}
			else
			{
				std::memset(hist_out,0,sizeof(uint32_t)*size*size*(2*steps*size*size+1));	
			}
			uint64_t total_size = size*size*(2*steps*size*size+1);
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t j=0;j<N_t;j++)
			{
				manage_thread_affinity();
				for(uint64_t i=0;i<total_size;i++)
				{
					#pragma omp atomic
					hist_out[i] += hist[j*total_size+i];
				}
			}
			return hist_out;
		}
		uint64_t getCount(){return count;}
		uint64_t getNbits(){return nbits;}
		uint64_t getSteps(){return steps;}
		uint64_t getSize(){return size;}
		uint64_t getThreads(){return N_t;}
};
