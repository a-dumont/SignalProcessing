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

// Gradient with full size x and t
template<class DataType, class DataType2>
void gradient(int n, DataType* x, DataType2* t, DataType* out)
{
	out[0] = (x[1]-x[0])/(t[1]-t[0]);
	out[n-1] = (x[n-1]-x[n-2])/(t[n-1]-t[n-2]);
	for (int i=1; i<(n-1); i++)
	{
			DataType hd = t[i+1]-t[i];
			DataType hs = t[i]-t[i-1];
			out[i] = (hs*hs*x[i+1]+(hd*hd-hs*hs)*x[i]-hd*hd*x[i-1])/(hs*hd*(hd+hs));
	}
}



// Gradient with fullsize x and constant dt
template<class DataType, class DataType2>
void gradient2(int n, DataType* x, DataType2 dt, DataType* out)
{
	DataType h = (DataType) 1/(2*dt);
	out[0] = 2*h*(x[1]-x[0]);
	out[n-1] = 2*h*(x[n-1]-x[n-2]);
	for (int i=1; i<(n-1); i++)
	{
			out[i] = h*(x[i+1]-x[i-1]);
	}
}

template<class DataType>
void finite_difference_coefficients(int M, int N, DataType* coeff)
{
	DataType alpha[2*N+1];
	N = 2*N;	
	alpha[0] = 0;
	alpha[1] = 1;
	DataType a; DataType b; DataType c;
	for(int i=2;i<(N+1);i++)
	{
		if(alpha[i-1] > 0)
		{
			alpha[i] = -alpha[i-1];
		}
		else
		{
			alpha[i] = -1*alpha[i-1]+1;
		}
	}
	coeff[0] = 1;
	a = 1;
	for(int n=1;n<(N+1);n++)
	{
		b = 1;
		for(int v=0;v<n;v++)
		{
			c = alpha[n]-alpha[v];
			b = b*c;
			for(int m=0;m<(std::min(M,n)+1);m++)
			{
				if (m != 0)
				{
					coeff[m*(N+1)*(N+1)+n*(N+1)+v] = (alpha[n]*coeff[m*(N+1)*(N+1)+(n-1)*(N+1)+v]-m*coeff[(m-1)*(N+1)*(N+1)+(n-1)*(N+1)+v])/c;
				}
				else
				{
					coeff[m*(N+1)*(N+1)+n*(N+1)+v] = (alpha[n]*coeff[m*(N+1)*(N+1)+(n-1)*(N+1)+v])/c;
				}
			}
		}
		for(int m=0;m<(std::min(M,n)+1);m++)
		{
			if (m != 0)
			{
				coeff[m*(N+1)*(N+1)+n*(N+1)+n] = a/b*(m*coeff[(m-1)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]-alpha[n-1]*coeff[(m)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]);
			}
			else
			{
				coeff[m*(N+1)*(N+1)+n*(N+1)+n] = -a/b*(alpha[n-1]*coeff[(m)*(N+1)*(N+1)+(n-1)*(N+1)+(n-1)]);
			}
		}
		a = b;
	}
}

template<class DataType, class DataType2>
void nth_order_gradient(int n, DataType* x,
				DataType2 dt, DataType* out, int M, int N, DataType* coeff)
{
	coeff += (M*(2*N+1)*(2*N+1)+2*N*(2*N+1));
	DataType norm = (DataType) 1.0/dt;
	int k;
	for(int i=N;i<(n-N);i++)
	{
		out[i-N] = coeff[0]*x[i];
		k = 1;
		for(int j=0;j<N;j++)
		{
			out[i-N] += coeff[k]*x[i+j+1]+coeff[k+1]*x[i-(j+1)];
			k += 2;
		}
		for(int l=0;l<M;l++){out[i-N] *= norm;}
	}
	coeff -= (M*(2*N+1)*(2*N+1)+2*N*(2*N+1));
}

template<class DataType>
void rolling_average(int n, DataType* in, DataType* out, int size)
{	
	DataType norm = (DataType) 1.0/size;
	for(int i=0;i<(n-size+1);i++)
	{	
		for(int j=0;j<size;j++)
		{
				out[i] += in[i+j];
		}
		out[i] *= norm;
	}
}

template<class DataType>
void continuous_max(long long int* out, DataType* in, int n)
{
	out[0] = 0;
	for(long long int i=1;i<n;i++)
	{
		if(in[i] > in[out[i-1]])
		{
			out[i] = i;
		}
		else 
		{
			out[i] = out[i-1];	
		}
	}
}

template<class DataType>
void continuous_min(long long int* out, DataType* in, int n)
{
	out[0] = 0;
	for(long long int i=1;i<n;i++)
	{
		if(in[i] < in[out[i-1]])
		{
			out[i] = i;
		}
		else 
		{
			out[i] = out[i-1];	
		}
	}
}

template<class DataType>
DataType sum_pairwise(DataType* in, long int n)
{
	if (n<=128)
	{
		if(n<8)
		{
			DataType res = 0;
			for(int i=0;i<n;i++)
			{
				res += in[i];
			}
			return res;
		}
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0;
			DataType out =  0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
			{
				res[0] += in[i];
				res[1] += in[i+1];
				res[2] += in[i+2]; 
				res[3] += in[i+3]; 
				res[4] += in[i+4]; 
				res[5] += in[i+5]; 
				res[6] += in[i+6]; 
				res[7] += in[i+7]; 
			}
			out = std::accumulate(res,res+8,remainder)+std::accumulate(in+N,in+n,remainder);
			return out;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
			{
				res[0] += in[128*j+i];
				res[1] += in[128*j+i+1];
				res[2] += in[128*j+i+2]; 
				res[3] += in[128*j+i+3]; 
				res[4] += in[128*j+i+4]; 
				res[5] += in[128*j+i+5]; 
				res[6] += in[128*j+i+6]; 
				res[7] += in[128*j+i+7]; 
			}
			out[j] = std::accumulate(res,res+8,remainder);
		}
		DataType res = sum_pairwise<DataType>(out,m)+std::accumulate(in+N,in+n,remainder);
		free(out);
		return res;
	}
}

template<class DataType>
DataType variance_pairwise(DataType* in, long int n)
{
	DataType _mean = sum_pairwise(in,n)/n;
	if (n<=128)
	{
		if(n<8)
		{
			DataType var = 0.0;
			for(int i=0;i<n;i++)
			{
				var += (in[i])*(in[i]);
			}
			return var/n-_mean*_mean;
		}
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0.0;
			DataType out =  0.0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
			{
				res[0] += (in[i])*(in[i]);
				res[1] += (in[i+1])*(in[i+1]);
				res[2] += (in[i+2])*(in[i+2]);
				res[3] += (in[i+3])*(in[i+3]); 
				res[4] += (in[i+4])*(in[i+4]);
				res[5] += (in[i+5])*(in[i+5]);
				res[6] += (in[i+6])*(in[i+6]); 
				res[7] += (in[i+7])*(in[i+7]);
			}
			out = std::accumulate(res,res+8,remainder);
			for(int i=N;i<n;i++)
			{
				remainder += (in[i])*(in[i]);
			}
			return (out+remainder)/n-_mean*_mean;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0.0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
			{
				res[0] += (in[128*j+i])*(in[128*j+i]);
				res[1] += (in[128*j+i+1])*(in[128*j+i+1]);
				res[2] += (in[128*j+i+2])*(in[128*j+i+2]);
				res[3] += (in[128*j+i+3])*(in[128*j+i+3]); 
				res[4] += (in[128*j+i+4])*(in[128*j+i+4]);
				res[5] += (in[128*j+i+5])*(in[128*j+i+5]);
				res[6] += (in[128*j+i+6])*(in[128*j+i+6]); 
				res[7] += (in[128*j+i+7])*(in[128*j+i+7]);
			}
			out[j] = std::accumulate(res,res+8,remainder);
		}
		for(int i=N;i<n;i++)
		{
			remainder += (in[i])*(in[i]);
		}
		DataType res = sum_pairwise<DataType>(out,m)+remainder;
		free(out);
		return res/n-_mean*_mean;
	}
}

template<class DataType>
DataType skewness_pairwise(DataType* in, long int n)
{
	DataType _mean = sum_pairwise(in,n)/n;
	if (n<=128)
	{
		if(n<8)
		{
			DataType skew = 0.0;
			for(int i=0;i<n;i++)
			{
				skew += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
			}
			return skew/n;
		}
		else 
		{
			long int N = n-n%8;
			DataType remainder = 0.0;
			DataType out =  0.0;
			DataType res[8] = {};
			for(long int i=0;i<N;i+=8)
			{
				res[0] += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
				res[1] += (in[i+1]-_mean)*(in[i+1]-_mean)*(in[i+1]-_mean);
				res[2] += (in[i+2]-_mean)*(in[i+2]-_mean)*(in[i+2]-_mean);
				res[3] += (in[i+3]-_mean)*(in[i+3]-_mean)*(in[i+3]-_mean); 
				res[4] += (in[i+4]-_mean)*(in[i+4]-_mean)*(in[i+4]-_mean);
				res[5] += (in[i+5]-_mean)*(in[i+5]-_mean)*(in[i+5]-_mean);
				res[6] += (in[i+6]-_mean)*(in[i+6]-_mean)*(in[i+6]-_mean); 
				res[7] += (in[i+7]-_mean)*(in[i+7]-_mean)*(in[i+7]-_mean);
			}
			out = std::accumulate(res,res+8,remainder);
			for(int i=N;i<n;i++)
			{
				remainder += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
			}
			return (out+remainder)/n;
		}
	}
	else
	{
		long int N = n-n%128;
		long int m = N/128;
		DataType remainder = 0.0;
		DataType* out = (DataType*) malloc(sizeof(DataType)*m);
		#pragma omp parallel for
		for(long int j=0;j<m;j++)
		{
			DataType res[8] = {};
			for(long int i=0;i<128;i+=8)
			{
				res[0] += (in[128*j+i]-_mean)*(in[128*j+i]-_mean)*(in[128*j+i]-_mean);
				res[1] += (in[128*j+i+1]-_mean)*(in[128*j+i+1]-_mean)*(in[128*j+i+1]-_mean);
				res[2] += (in[128*j+i+2]-_mean)*(in[128*j+i+2]-_mean)*(in[128*j+i+2]-_mean);
				res[3] += (in[128*j+i+3]-_mean)*(in[128*j+i+3]-_mean)*(in[128*j+i+3]-_mean); 
				res[4] += (in[128*j+i+4]-_mean)*(in[128*j+i+4]-_mean)*(in[128*j+i+4]-_mean);
				res[5] += (in[128*j+i+5]-_mean)*(in[128*j+i+5]-_mean)*(in[128*j+i+5]-_mean);
				res[6] += (in[128*j+i+6]-_mean)*(in[128*j+i+6]-_mean)*(in[128*j+i+6]-_mean); 
				res[7] += (in[128*j+i+7]-_mean)*(in[128*j+i+7]-_mean)*(in[128*j+i+7]-_mean);
			}
			out[j] = std::accumulate(res,res+8,remainder);
		}
		for(int i=N;i<n;i++)
		{
			remainder += (in[i]-_mean)*(in[i]-_mean)*(in[i]-_mean);
		}
		DataType res = sum_pairwise<DataType>(out,m)+remainder;
		free(out);
		return res/n;
	}
}

template<class DataType, class DataType2>
void product(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]*in2[i];
	}
}

template<class DataType, class DataType2>
void sum(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]+in2[i];
	}
}

template<class DataType, class DataType2>
void difference(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]-in2[i];
	}
}

template<class DataType, class DataType2>
void division(DataType* in1, DataType* in2, DataType2* out, int n)
{
	#pragma omp parallel for
	for(int i=0;i<n;i++)
	{
		out[i] = in1[i]/in2[i];
	}
}

template<class DataType>
DataType max(DataType* in, int n)
{
	DataType _max = in[0];
	#pragma omp parallel for default(shared) reduction(max:_max)
	for(int i=1;i<n;i++)
	{
		_max = _max > in[i] ? _max : in[i];
	}
	return _max;
}

template<class DataType>
DataType min(DataType* in, int n)
{
	DataType _min = in[0];
	#pragma omp parallel for default(shared) reduction(min:_min)
	for(int i=1;i<n;i++)
	{
		_min = _min < in[i] ? _min : in[i];
	}
	return _min;
}

template<class DataTypeIn, class DataTypeOut>
void block_max(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out)
{
	int64_t n = N/block_size;;
		
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		out[i] = (DataTypeOut) *std::max_element(in+i*block_size,in+(i+1)*block_size);
	}
}

template<class DataTypeIn, class DataTypeOut>
void block_min(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out)
{
	int64_t n = N/block_size;
		
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		out[i] = (DataTypeOut) *std::min_element(in+i*block_size,in+(i+1)*block_size);
	}
}

template<class DataTypeIn, class DataTypeOut>
void block_min_max(int64_t N, int64_t block_size, DataTypeIn* in, DataTypeOut* out)
{
	int64_t n = N/block_size;
		
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		out[i] = (DataTypeOut) *std::min_element(in+i*block_size,in+(i+1)*block_size);
		out[i+N] = (DataTypeOut) *std::max_element(in+i*block_size,in+(i+1)*block_size);
	}
}

class DigitizerBlockMax
{
	protected:
		int64_t N, max_size, min_size, n_max, n_min, resolution;
		int64_t hist_size = 0;
		uint64_t* buffer;
		uint64_t* buffer2;
		uint64_t* max_hists;
		int64_t count = 0;
	
	public:
		DigitizerBlockMax(int64_t N_in, int64_t min_size_in, 
						int64_t max_size_in, int64_t resolution_in)
		{
			if (max_size_in%2 != 0 )
			{
				throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			if (min_size_in%2 != 0 )
			{
					throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			N = N_in;
			resolution = resolution_in;
			max_size = max_size_in;
			min_size = min_size_in;
			n_max = N_in/min_size_in;
			n_min = N_in/max_size_in;
			hist_size = (int64_t) (log2(n_max/n_min)+1)*(1<<resolution_in);
			buffer = (uint64_t*) malloc(sizeof(uint64_t)*n_max);
			buffer2 = (uint64_t*) malloc(sizeof(uint64_t)*n_max/2);
			max_hists = (uint64_t*) malloc(sizeof(uint64_t)*hist_size);
			std::memset(max_hists,0,sizeof(uint64_t)*hist_size);
		}
		
		~DigitizerBlockMax()
		{
			free(buffer);
			free(buffer2);
			free(max_hists);
		}
		
		template<class DataType>
		void accumulate(DataType* in)
		{
			block_max<DataType,uint64_t>(N,min_size,in,buffer);
			//#pragma omp parallel for reduction(+:max_hists[:1<<resolution])
			for(int64_t i=0;i<n_max;i++)
			{
				max_hists[buffer[i]] += 1;
			}
			recursion(n_max,2*min_size,max_hists+(1<<resolution),buffer,buffer2);
			count += 1;
		}
		
		void recursion(int64_t N_in, int64_t block_size, uint64_t* out, 
						uint64_t* buf_in, uint64_t* buf_out)
		{
			if(block_size < max_size)
			{
				block_max(N_in,2,buf_in,buf_out);
				//#pragma omp parallel for reduction(+:out[:1<<resolution])
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out[buf_out[i]] += 1;
				}
				recursion(N_in/2,block_size*2,out+(1<<resolution),buf_out,buf_in);
			}
			else 
			{
				//out[std::max(buf_in[0],buf_in[1])] += 1;
				block_max(N_in,2,buf_in,buf_out);
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out[buf_out[i]] += 1;
				}
			}
		}
		
		uint64_t* get_max_hists(){return max_hists;}
		int64_t get_resolution(){return resolution;}
		int64_t get_N(){return N;}
		int64_t get_min_size(){return min_size;}
		int64_t get_max_size(){return max_size;}
		int64_t get_count(){return count;}
		void clear()
		{
			std::memset(max_hists,0,sizeof(uint64_t)*hist_size);
			std::memset(buffer,0,sizeof(uint64_t)*n_max);
			std::memset(buffer2,0,sizeof(uint64_t)*n_max/2);
			count = 0;
		}
};

class DigitizerBlockMin
{
	protected:
		int64_t N, max_size, min_size, n_max, n_min, resolution;
		int64_t hist_size = 0;
		uint64_t* buffer;
		uint64_t* buffer2;
		uint64_t* min_hists;
		int64_t count = 0;
	
	public:
		DigitizerBlockMin(int64_t N_in, int64_t min_size_in, 
						int64_t max_size_in, int64_t resolution_in)
		{
			if (max_size_in%2 != 0 )
			{
				throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			if (min_size_in%2 != 0 )
			{
					throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			N = N_in;
			resolution = resolution_in;
			max_size = max_size_in;
			min_size = min_size_in;
			n_max = N_in/min_size_in;
			n_min = N_in/max_size_in;
			hist_size = (int64_t) (log2(n_max/n_min)+1)*(1<<resolution_in);
			buffer = (uint64_t*) malloc(sizeof(uint64_t)*n_max);
			buffer2 = (uint64_t*) malloc(sizeof(uint64_t)*n_max/2);
			min_hists = (uint64_t*) malloc(sizeof(uint64_t)*hist_size);
			std::memset(min_hists,0,sizeof(uint64_t)*hist_size);
		}
		
		~DigitizerBlockMin()
		{
			free(buffer);
			free(buffer2);
			free(min_hists);
		}
		
		template<class DataType>
		void accumulate(DataType* in)
		{
			block_min<DataType,uint64_t>(N,min_size,in,buffer);
			//#pragma omp parallel for reduction(+:max_hists[:1<<resolution])
			for(int64_t i=0;i<n_max;i++)
			{
				min_hists[buffer[i]] += 1;
			}
			recursion(n_max,2*min_size,min_hists+(1<<resolution),buffer,buffer2);
			count += 1;
		}
		
		void recursion(int64_t N_in, int64_t block_size, uint64_t* out, 
						uint64_t* buf_in, uint64_t* buf_out)
		{
			if(block_size < max_size)
			{
				block_min(N_in,2,buf_in,buf_out);
				//#pragma omp parallel for reduction(+:out[:1<<resolution])
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out[buf_out[i]] += 1;
				}
				recursion(N_in/2,block_size*2,out+(1<<resolution),buf_out,buf_in);
			}
			else 
			{
				//out[std::min(buf_in[0],buf_in[1])] += 1;
				block_min(N_in,2,buf_in,buf_out);
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out[buf_out[i]] += 1;
				}
			}
		}
		
		uint64_t* get_min_hists(){return min_hists;}
		int64_t get_resolution(){return resolution;}
		int64_t get_N(){return N;}
		int64_t get_min_size(){return min_size;}
		int64_t get_max_size(){return max_size;}
		int64_t get_count(){return count;}
		void clear()
		{
			std::memset(min_hists,0,sizeof(uint64_t)*hist_size);
			std::memset(buffer,0,sizeof(uint64_t)*n_max);
			std::memset(buffer2,0,sizeof(uint64_t)*n_max/2);
			count = 0;
		}
};

class DigitizerBlockMinMax
{
	protected:
		int64_t N, max_size, min_size, n_max, n_min, resolution;
		int64_t hist_size = 0;
		uint64_t* buffer;
		uint64_t* buffer2;
		uint64_t* buffer3;
		uint64_t* buffer4;
		uint64_t* min_hists;
		uint64_t* max_hists;
		int64_t count = 0;
	
	public:
		DigitizerBlockMinMax(int64_t N_in, int64_t min_size_in, 
						int64_t max_size_in, int64_t resolution_in)
		{
			if (max_size_in%2 != 0 )
			{
				throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			if (min_size_in%2 != 0 )
			{
					throw std::runtime_error("U dumbdumb, block size must be power of 2.");
			}
			N = N_in;
			resolution = resolution_in;
			max_size = max_size_in;
			min_size = min_size_in;
			n_max = N_in/min_size_in;
			n_min = N_in/max_size_in;
			hist_size = (int64_t) (log2(n_max/n_min)+1)*(1<<resolution_in);
			buffer = (uint64_t*) malloc(sizeof(uint64_t)*n_max);
			buffer2 = (uint64_t*) malloc(sizeof(uint64_t)*n_max/2);
			buffer3 = (uint64_t*) malloc(sizeof(uint64_t)*n_max);
			buffer4 = (uint64_t*) malloc(sizeof(uint64_t)*n_max/2);
			min_hists = (uint64_t*) malloc(sizeof(uint64_t)*hist_size);
			max_hists = (uint64_t*) malloc(sizeof(uint64_t)*hist_size);
			std::memset(min_hists,0,sizeof(uint64_t)*hist_size);
			std::memset(max_hists,0,sizeof(uint64_t)*hist_size);
		}
		
		~DigitizerBlockMinMax()
		{
			free(buffer);
			free(buffer2);
			free(buffer3);
			free(buffer4);
			free(min_hists);
			free(max_hists);
		}
		
		template<class DataType>
		void accumulate(DataType* in)
		{
			block_min<DataType,uint64_t>(N,min_size,in,buffer);
			block_max<DataType,uint64_t>(N,min_size,in,buffer3);
			//#pragma omp parallel for reduction(+:max_hists[:1<<resolution])
			for(int64_t i=0;i<n_max;i++)
			{
				min_hists[buffer[i]] += 1;
				max_hists[buffer3[i]] += 1;
			}
			recursion(n_max,2*min_size,min_hists+(1<<resolution),max_hists+(1<<resolution),
							buffer,buffer2,buffer3,buffer4);
			count += 1;
		}
		
		void recursion(int64_t N_in, int64_t block_size, uint64_t* out_min, uint64_t* out_max, 
						uint64_t* buf_in_min, uint64_t* buf_out_min, uint64_t* buf_in_max, 
						uint64_t* buf_out_max)
		{
			if(block_size < max_size)
			{
				block_min(N_in,2,buf_in_min,buf_out_min);
				block_max(N_in,2,buf_in_max,buf_out_max);
				//#pragma omp parallel for reduction(+:out[:1<<resolution])
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out_min[buf_out_min[i]] += 1;
					out_max[buf_out_max[i]] += 1;
				}
				recursion(N_in/2,block_size*2,out_min+(1<<resolution),out_max+(1<<resolution), 
								buf_out_min,buf_in_min,buf_out_max,buf_in_max);
			}
			else 
			{
				//out_min[std::min(buf_in_min[0],buf_in_min[1])] += 1;
				//out_max[std::max(buf_in_max[0],buf_in_max[1])] += 1;
				block_min(N_in,2,buf_in_min,buf_out_min);
				block_max(N_in,2,buf_in_max,buf_out_max);
				for(int64_t i=0;i<(N_in/2);i++)
				{
					out_min[buf_out_min[i]] += 1;
					out_max[buf_out_max[i]] += 1;
				}
			}
		}
		
		uint64_t* get_min_hists(){return min_hists;}
		uint64_t* get_max_hists(){return min_hists;}
		int64_t get_resolution(){return resolution;}
		int64_t get_N(){return N;}
		int64_t get_min_size(){return min_size;}
		int64_t get_max_size(){return max_size;}
		int64_t get_count(){return count;}
		void clear()
		{
			std::memset(min_hists,0,sizeof(uint64_t)*hist_size);
			std::memset(max_hists,0,sizeof(uint64_t)*hist_size);
			std::memset(buffer,0,sizeof(uint64_t)*n_max);
			std::memset(buffer2,0,sizeof(uint64_t)*n_max/2);
			std::memset(buffer3,0,sizeof(uint64_t)*n_max);
			std::memset(buffer4,0,sizeof(uint64_t)*n_max/2);
			count = 0;
		}
};

