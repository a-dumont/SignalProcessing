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

	#pragma omp parallel for
	for(int i=steps;i<(n-steps);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			int xbin = std::clamp((int)((xdata[i]-xmin)*xstep_inv),0,nbins-1);
			int ybin = std::clamp((int)((ydata[i]-ymin)*ystep_inv),0,nbins-1);
			#pragma omp atomic
			hist_after[ybin+nbins*xbin] += 1;

			for(int j=1;j<steps+1;j++)
			{
				int xbin2 = std::clamp((int)((xdata[i+j]-xmin)*xstep_inv),0,nbins-1);
				int ybin2 = std::clamp((int)((ydata[i+j]-ymin)*ystep_inv),0,nbins-1);
				int xbin3 = std::clamp((int)((xdata[i-j]-xmin)*xstep_inv),0,nbins-1);
				int ybin3 = std::clamp((int)((ydata[i-j]-ymin)*ystep_inv),0,nbins-1);
				#pragma omp atomic
				hist_after[nbins*nbins+(xbin*nbins*nbins*nbins+ybin*nbins*nbins)+nbins*xbin2+ybin2+(j-1)*nbins*nbins*nbins*nbins] += 1;
				#pragma omp atomic
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
	for(long long int i=0;i<nbins;i++)
	{
		for(long long int j=0;j<nbins;j++)
		{
			out[0] += hist[i*nbins+j]*(i-row);
			out[1] += hist[i*nbins+j]*(j-col);
		}
	}
}

template<class DataType>
void histogram_nth_order_derivative(long long int nbins, DataType* data_after, DataType* data_before, DataType dt, long long int m, long long int n, DataType* out, DataType* coeff)
{
	long long int size = nbins*nbins*nbins*nbins;
	DataType in[2*n+1];
	in[n] = 0;
	#pragma omp parallel for
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
