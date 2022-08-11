#include <stdexcept>

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

template <class DataType>
void GetEdges(DataType* data, long long int n, long long int nbins, DataType* edges)
{
	std::pair<DataType*,DataType*> minmax = std::minmax_element(data,data+n);
	DataType min = *minmax.first;
	DataType max = *minmax.second;
	DataType step = (DataType) (max-min)/nbins;
	edges[0] = min;
	edges[nbins] = max;
	for(long long int i=1;i<(nbins);i++)
	{
		edges[i] = edges[i-1]+step;
	}
}

template <class DataType>
void Histogram(long long int* hist, DataType* edges, DataType* data,
				long long int n, long long int nbins)
{	
	double step_inv = 1.0/(edges[1]-edges[0]);
	DataType min = edges[0];
	DataType max = edges[nbins];
	long long int zero = 0;
	for(long long int i=0;i<n;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			long long int bin = (long long int)((data[i]-min)*step_inv);
			bin = std::clamp(bin,zero,nbins-1);
			hist[bin] += 1;
		}
	}
}

template <class DataType>
void Histogram_Density(DataType* hist, DataType* edges,
				DataType* data, long long int n, long long int nbins)
{	
	DataType step_inv = 1.0/(edges[1]-edges[0]);
	DataType norm = step_inv/n;
	DataType min = edges[0];
	DataType max = edges[nbins];
	long long int zero = 0;
	for(long long int i=0;i<n;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			long long int bin = (long long int)((data[i]-min)*step_inv);
			bin = std::clamp(bin,zero,nbins-1);
			hist[bin] += norm;
		}
	}
}

template <class DataType>
void Histogram_2D(long long int* hist, DataType* xedges, DataType* yedges,
				DataType* xdata, DataType* ydata, long long int n, long long int nbins)
{	
	double xstep_inv = 1.0/(xedges[1]-xedges[0]);
	double ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	long long int zero = 0;
	for(long long int i=0;i<n;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				long long int xbin = (long long int)((xdata[i]-xmin)*xstep_inv);
				long long int ybin = (long long int)((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,zero,nbins-1);
				ybin = std::clamp(ybin,zero,nbins-1);
				hist[ybin+nbins*xbin] += 1;
		}
	}
}

template <class DataType>
void Histogram_2D_Density(DataType* hist, DataType* xedges, DataType* yedges, DataType* xdata, DataType* ydata, long long int n, long long int nbins)
{
	DataType xstep_inv = 1.0/(xedges[1]-xedges[0]);
	DataType ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	DataType norm = xstep_inv*ystep_inv/n;
	long long int zero = 0;
	for(long long int i=0;i<n;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				long long int xbin = (long long int)((xdata[i]-xmin)*xstep_inv);
				long long int ybin = (long long int)((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,zero,nbins-1);
				ybin = std::clamp(ybin,zero,nbins-1);
				hist[ybin+nbins*xbin] += norm;
		}
	}
}


template <class DataType>
long long int Find_First_In_Bin(DataType* data, DataType* edges, long long int n)
{
	for(long long int i=0;i<n;i++)
	{
		if(data[i]>=edges[0] && data[i] <= edges[1])
		{
			return (long long int) i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template <class DataType>
long long int Find_First_In_Bin_2D(DataType* xdata, DataType* ydata, DataType* xedges, DataType* yedges, long long int n)
{
	for(long long int i=0;i<n;i++)
	{
		if(xdata[i]>=xedges[0] && xdata[i] <= xedges[1] && ydata[i] >= yedges[0] && ydata[i] <= yedges[1])
		{
			return (long long int) i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template<class DataType>
void Histogram_And_Displacement_2D(uint64_t* hist, DataType* xedges, DataType* yedges, DataType* xdata, DataType* ydata, long long int n, long long int nbins)
{	
	double xstep_inv = 1.0/(xedges[1]-xedges[0]);
	double ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	long long int zero = (long long int) 0;

	#pragma omp parallel for
	for(long long int i=0;i<(n-1);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			long long int xbin = (long long int)((xdata[i]-xmin)*xstep_inv);
			long long int ybin = (long long int)((ydata[i]-ymin)*ystep_inv);
			long long int xbin2 = (long long int)((xdata[i+1]-xmin)*xstep_inv);
			long long int ybin2 = (long long int)((ydata[i+1]-ymin)*ystep_inv);
			xbin = std::clamp(xbin,zero,nbins-1);
			ybin = std::clamp(ybin,zero,nbins-1);
			xbin2 = std::clamp(xbin2,zero,nbins-1);
			ybin2 = std::clamp(ybin2,zero,nbins-1);
			#pragma omp atomic
			hist[ybin+nbins*xbin] += 1;
			#pragma omp atomic
			hist[nbins*nbins+xbin*nbins*nbins*nbins+ybin*nbins*nbins+nbins*xbin2+ybin2] += 1;
		}
	}
	if( ((xdata[n]-xmax)*(xdata[n]-xmin) <= 0) && ((ydata[n]-ymax)*(ydata[n]-ymin) <= 0) )
	{	
		long long int xbin = (long long int)((xdata[n]-xmin)*xstep_inv);
		long long int ybin = (long long int)((ydata[n]-ymin)*ystep_inv);
		xbin = std::clamp(xbin,zero,nbins-1);
		ybin = std::clamp(ybin,zero,nbins-1);
		hist[ybin+nbins*xbin] += 1;
	}
}

template<class DataType>
void Histogram_And_Displacement_2D_steps(uint64_t* hist_after, uint64_t* hist_before, DataType* xedges, DataType* yedges, DataType* xdata, DataType* ydata, int n, int nbins,int steps)
{	
	DataType xstep_inv = 1/(xedges[1]-xedges[0]);
	DataType ystep_inv = 1/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];

	//#pragma omp parallel for
	for(int i=steps;i<(n-steps);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			int xbin = std::clamp((int)((xdata[i]-xmin)*xstep_inv),0,nbins-1);
			int ybin = std::clamp((int)((ydata[i]-ymin)*ystep_inv),0,nbins-1);
			//#pragma omp atomic
			hist_after[ybin+nbins*xbin] += 1;

			for(int j=1;j<steps+1;j++)
			{
				int xbin2 = std::clamp((int)((xdata[i+j]-xmin)*xstep_inv),0,nbins-1);
				int ybin2 = std::clamp((int)((ydata[i+j]-ymin)*ystep_inv),0,nbins-1);
				int xbin3 = std::clamp((int)((xdata[i-j]-xmin)*xstep_inv),0,nbins-1);
				int ybin3 = std::clamp((int)((ydata[i-j]-ymin)*ystep_inv),0,nbins-1);
				//#pragma omp atomic
				hist_after[nbins*nbins+(xbin*nbins*nbins*nbins+ybin*nbins*nbins)+nbins*xbin2+ybin2+(j-1)*nbins*nbins*nbins*nbins] += 1;
				//#pragma omp atomic
				hist_before[(xbin*nbins*nbins*nbins+ybin*nbins*nbins)+nbins*xbin3+ybin3+(j-1)*nbins*nbins*nbins*nbins] += 1;
			}
		}
	}
}


template<class DataType>
class cHistogram2D 
{
	protected:
		long long int* hist;
		DataType* xedges;
		DataType* yedges;
		long long int nbins;
		long long int n;
		long long int count;
	public:
		cHistogram2D(DataType* xdata, DataType* ydata, long long int Nbins, long long int N)
		{
			nbins = Nbins;
			hist = (long long int*) malloc(sizeof(long long int)*nbins*nbins);
			std::memset(hist,0,sizeof(long)*nbins*nbins);
			n = N;
			xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(xdata, n, nbins,xedges);
			yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(ydata, n, nbins,yedges);
			Histogram_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		~cHistogram2D(){free(hist);free(xedges);free(yedges);}
		void accumulate(DataType* xdata, DataType* ydata)
		{
			Histogram_2D<DataType>(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		long long int* getHistogram(){return hist;}
		long long int getCount(){return count;}
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		long long int getNbins(){return nbins;}
};

template<class DataType>
class cHistogram_2D_Density 
{
	protected:
		DataType* hist;  
		DataType* xedges;
		DataType* yedges;
		long long int nbins;
		long long int n;
		long long int count;
	public:
		cHistogram_2D_Density(DataType* xdata, DataType* ydata,
						long long int Nbins, long long int N)
		{
			nbins = Nbins;
			n = N;
			hist = (DataType*) malloc(sizeof(DataType)*nbins*nbins);
			hist = (DataType*) std::memset(hist,0,sizeof(DataType)*nbins*nbins);
			xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(xdata, n, nbins,xedges);
			yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(ydata, n, nbins,yedges);
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		~cHistogram_2D_Density(){free(hist);free(xedges);free(yedges);}
		void accumulate(DataType* xdata, DataType* ydata)
		{
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		DataType* getHistogram(){return hist;}
		long long int getCount(){return count;}
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		long long int getNbins(){return nbins;}
};

template<class DataType>
class cHistogram_And_Displacement_2D 
{
	protected:
		uint64_t* hist;
		DataType* xedges;
		DataType* yedges;
		long long int nbins;
		long long int n;
		long long int count;
	public:
		cHistogram_And_Displacement_2D(DataType* xdata, DataType* ydata,
						long long int Nbins, long long int N)
		{
			nbins = Nbins;
			n = N;
			hist = (uint64_t*) malloc(sizeof(uint64_t)*(nbins*nbins*(nbins*nbins+1)));
			hist = (uint64_t*) std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(xdata, n, nbins,xedges);
			yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(ydata, n, nbins,yedges);
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		~cHistogram_And_Displacement_2D(){free(hist);free(xedges);free(yedges);}
		void accumulate(DataType* xdata, DataType* ydata)
		{
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		uint64_t* getHistogram(){return hist;}
		long long int getCount(){return count;}
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		long long int getNbins(){return nbins;}
};

template<class DataType>
class cHistogram_And_Displacement_2D_steps 
{
	protected:
		uint64_t* hist;
		uint64_t* hist_before;
		DataType* xedges;
		DataType* yedges;
		long long int nbins;
		long long int n;
		long long int count;
		long long int steps;
	public:
		cHistogram_And_Displacement_2D_steps(DataType* xdata, DataType* ydata,
						long long int Nbins, long long int N, int long long Steps)
		{
			nbins = Nbins;
			n = N;
			steps = Steps;
			hist = (uint64_t*) malloc(sizeof(uint64_t)*(nbins*nbins*(2*steps*nbins*nbins+1)));
			hist = (uint64_t*) std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(2*steps*nbins*nbins+1));
			hist_before = hist+((nbins*nbins)*(steps*nbins*nbins+1));
			xedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(xdata, n, nbins,xedges);
			yedges = (DataType*) malloc(sizeof(DataType)*(nbins+1));
			GetEdges(ydata, n, nbins,yedges);
			Histogram_And_Displacement_2D_steps(hist,hist_before,xedges,yedges,xdata,ydata,n,nbins,steps);
			count = 1;
		}
		~cHistogram_And_Displacement_2D_steps(){free(hist);free(xedges);free(yedges);}
		void accumulate(DataType* xdata, DataType* ydata)
		{
			Histogram_And_Displacement_2D_steps(hist,hist_before,xedges,yedges,xdata,ydata,n,nbins,steps);
			count += 1;
		}
		uint64_t* getHistogram(){return hist;}
		long long int getCount(){return count;}
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		long long int getNbins(){return nbins;}
};

template<class DataType>
void histogram_vectorial_average(long long int nbins, 
				DataType* hist, DataType* out, long long int row, long long int col)
{
	double norm;
	#pragma omp parallel for reduction(+:out[:2])
	for(long long int i=0;i<nbins;i++)
	{
		for(long long int j=0;j<nbins;j++)
		{
			norm = sqrt((i-row)*(i-row)+(j-col)*(j-col));
			out[0] += hist[i*nbins+j]*(i-row)/norm;
			out[1] += hist[i*nbins+j]*(j-col)/norm;
		}
	}
}

template<class DataType>
void histogram_nth_order_derivative(long long int nbins, DataType* data_after, DataType* data_before, DataType dt, long long int m, long long int n, DataType* out, DataType* coeff)
{
	long long int size = nbins*nbins*nbins*nbins;
	DataType in[2*n+1];
	in[n] = 0;
	//#pragma omp parallel for
	for(long long int i=0;i<nbins;i++)
	{
		for(long long int j=0;j<nbins;j++)
		{
			for(long long int k=0;k<nbins;k++)
			{
				for(long long int l=0;l<nbins;l++)
				{
					for(int s=0;s<n;s++)
					{
						in[n+1+s]=data_after[s*size+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l];
						in[n-1-s]=data_before[s*size+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l];
					}
					nth_order_gradient<DataType,DataType>((2*n)+1,
									in,dt,
									out+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l,
									m,n,coeff);
				}
			}
		}
	}
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
                N_t = std::min(64,omp_get_max_threads()*nbgroups);
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
