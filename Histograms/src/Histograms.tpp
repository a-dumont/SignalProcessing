template <class Datatype>
Datatype* GetEdges(Datatype* data, int n, int nbins)
{
	Datatype* edges = (Datatype*) malloc(sizeof(Datatype*)*(nbins+1));
	std::pair<Datatype*,Datatype*> minmax = std::minmax_element(data,data+n);
	Datatype min = *minmax.first;
	Datatype max = *minmax.second;
	Datatype step = (max-min)/nbins;
	edges[0] = min;
	edges[nbins] = max;
	for(int i=1;i<(nbins);i++)
	{
		edges[i] = edges[i-1]+step;
	}
	return edges;
}

template <class Datatype>
void Histogram(long* hist,Datatype* edges, Datatype* data, int n, int nbins)
{	
	Datatype step_inv = 1/(edges[1]-edges[0]);
	Datatype min = edges[0];
	Datatype max = edges[nbins];	
	for(int i=0;i<n;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			int bin = (int)((data[i]-min)*step_inv);
			bin = std::clamp(bin,0,nbins-1);
			hist[bin] += 1;
		}
	}
}

template <class Datatype>
void Histogram_Density(Datatype* hist,Datatype* edges, Datatype* data, int n, int nbins)
{	
	for(int i=0;i<nbins;i++)
	{
		hist[i] = 0;
	}
	Datatype step_inv = 1/(edges[1]-edges[0]);
	Datatype norm = step_inv/n;
	Datatype min = edges[0];
	Datatype max = edges[nbins];	
	for(int i=0;i<n;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			int bin = (int)((data[i]-min)*step_inv);
			bin = std::clamp(bin,0,nbins-1);
			hist[bin] += norm;
		}
	}
}

template <class Datatype>
void Histogram_2D(long* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
{	
	Datatype xstep_inv = 1/(xedges[1]-xedges[0]);
	Datatype ystep_inv = 1/(yedges[1]-yedges[0]);
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];
	for(int i=0;i<n;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				int xbin = (int)((xdata[i]-xmin)*xstep_inv);
				int ybin = (int)((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,0,nbins-1);
				ybin = std::clamp(ybin,0,nbins-1);
				hist[ybin+nbins*xbin] += 1;
		}
	}
}

template <class Datatype>
void Histogram_2D_Density(double* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
{
	Datatype xstep_inv = 1/(xedges[1]-xedges[0]);
	Datatype ystep_inv = 1/(yedges[1]-yedges[0]);
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];
	Datatype norm = xstep_inv*ystep_inv/n;
	for(int i=0;i<n;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				int xbin = (int)((xdata[i]-xmin)*xstep_inv);
				int ybin = (int)((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,0,nbins-1);
				ybin = std::clamp(ybin,0,nbins-1);
				hist[ybin+nbins*xbin] += norm;
		}
	}
}


template <class Datatype, class Datatype2>
int Find_First_In_Bin(Datatype* data, Datatype2* edges, int n)
{
	for(int i=0;i<n;i++)
	{
		if(data[i]>=edges[0] && data[i] <= edges[1])
		{
			return i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template <class Datatype, class Datatype2>
int Find_First_In_Bin_2D(Datatype* xdata, Datatype* ydata, Datatype2* xedges, Datatype2* yedges, int n)
{
	for(int i=0;i<n;i++)
	{
		if(xdata[i]>=xedges[0] && xdata[i] <= xedges[1] && ydata[i] >= yedges[0] && ydata[i] <= yedges[1])
		{
			return i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template<class Datatype>
void Histogram_And_Displacement_2D(uint64_t* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
{	
	Datatype xstep_inv = 1/(xedges[1]-xedges[0]);
	Datatype ystep_inv = 1/(yedges[1]-yedges[0]);
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];

	#pragma omp parallel for
	for(int i=0;i<(n-1);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			int xbin = (int)((xdata[i]-xmin)*xstep_inv);
			int ybin = (int)((ydata[i]-ymin)*ystep_inv);
			int xbin2 = (int)((xdata[i+1]-xmin)*xstep_inv);
			int ybin2 = (int)((ydata[i+1]-ymin)*ystep_inv);
			xbin = std::clamp(xbin,0,nbins-1);
			ybin = std::clamp(ybin,0,nbins-1);
			xbin2 = std::clamp(xbin2,0,nbins-1);
			ybin2 = std::clamp(ybin2,0,nbins-1);
			#pragma omp atomic
			hist[ybin+nbins*xbin] += 1;
			#pragma omp atomic
			hist[nbins*nbins+xbin*nbins*nbins*nbins+ybin*nbins*nbins+nbins*xbin2+ybin2] += 1;
		}
	}
	if( ((xdata[n]-xmax)*(xdata[n]-xmin) <= 0) && ((ydata[n]-ymax)*(ydata[n]-ymin) <= 0) )
	{	
		int xbin = (int)((xdata[n]-xmin)*xstep_inv);
		int ybin = (int)((ydata[n]-ymin)*ystep_inv);
		xbin = std::clamp(xbin,0,nbins-1);
		ybin = std::clamp(ybin,0,nbins-1);
		hist[ybin+nbins*xbin] += 1;
	}
}

template<class Datatype>
void Histogram_And_Displacement_2D_steps(uint64_t* hist_after, uint64_t* hist_before, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins,int steps)
{	
	Datatype xstep_inv = 1/(xedges[1]-xedges[0]);
	Datatype ystep_inv = 1/(yedges[1]-yedges[0]);
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];

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


template<class Datatype>
class cHistogram2D 
{
	protected:
		long* hist;
		Datatype* xedges;
		Datatype* yedges;
		int nbins;
		int n;
		int count;
	public:
		cHistogram2D(Datatype* xdata, Datatype* ydata, int Nbins, int N)
		{
			nbins = Nbins;
			hist = (long*) malloc(sizeof(long)*nbins*nbins);
			hist = (long*) std::memset(hist,0,sizeof(long)*nbins*nbins);
			n = N;
			xedges = GetEdges(xdata, n, nbins);
			yedges = GetEdges(ydata, n, nbins);
			Histogram_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		void accumulate(Datatype* xdata, Datatype* ydata)
		{
			Histogram_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		long* getHistogram(){return hist;}
		int getCount(){return count;}
		std::tuple<Datatype*,Datatype*> getEdges(){return std::make_tuple(xedges,yedges);}
		int getNbins(){return nbins;}
};

template<class Datatype>
class cHistogram_2D_Density 
{
	protected:
		double* hist;  
		Datatype* xedges;
		Datatype* yedges;
		int nbins;
		int n;
		int count;
	public:
		cHistogram_2D_Density(Datatype* xdata, Datatype* ydata, int Nbins, int N)
		{
			nbins = Nbins;
			n = N;
			hist = (double*) malloc(sizeof(double)*nbins*nbins);
			hist = (double*) std::memset(hist,0,sizeof(double)*nbins*nbins);
			xedges = GetEdges(xdata, n, nbins);
			yedges = GetEdges(ydata, n, nbins);
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		void accumulate(Datatype* xdata, Datatype* ydata)
		{
			Histogram_2D_Density(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		double* getHistogram(){return hist;}
		int getCount(){return count;}
		std::tuple<Datatype*,Datatype*> getEdges(){return std::make_tuple(xedges,yedges);}
		int getNbins(){return nbins;}
};

template<class Datatype>
class cHistogram_And_Displacement_2D 
{
	protected:
		uint64_t* hist;
		Datatype* xedges;
		Datatype* yedges;
		int nbins;
		int n;
		int count;
	public:
		cHistogram_And_Displacement_2D(Datatype* xdata, Datatype* ydata, int Nbins, int N)
		{
			nbins = Nbins;
			n = N;
			hist = (uint64_t*) malloc(sizeof(uint64_t)*(nbins*nbins*(nbins*nbins+1)));
			hist = (uint64_t*) std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(nbins*nbins+1));
			xedges = GetEdges(xdata, n, nbins);
			yedges = GetEdges(ydata, n, nbins);
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count = 1;
		}
		void accumulate(Datatype* xdata, Datatype* ydata)
		{
			Histogram_And_Displacement_2D(hist,xedges,yedges,xdata,ydata,n,nbins);
			count += 1;
		}
		uint64_t* getHistogram(){return hist;}
		int getCount(){return count;}
		std::tuple<Datatype*,Datatype*> getEdges(){return std::make_tuple(xedges,yedges);}
		int getNbins(){return nbins;}
};

template<class Datatype>
class cHistogram_And_Displacement_2D_steps 
{
	protected:
		uint64_t* hist;
		uint64_t* hist_before;
		Datatype* xedges;
		Datatype* yedges;
		int nbins;
		int n;
		int count;
		int steps;
	public:
		cHistogram_And_Displacement_2D_steps(Datatype* xdata, Datatype* ydata, int Nbins, int N, int Steps)
		{
			nbins = Nbins;
			n = N;
			steps = Steps;
			hist = (uint64_t*) malloc(sizeof(uint64_t)*(nbins*nbins*(2*steps*nbins*nbins+1)));
			hist = (uint64_t*) std::memset(hist,0,sizeof(uint64_t)*nbins*nbins*(2*steps*nbins*nbins+1));
			hist_before = hist+((nbins*nbins)*(steps*nbins*nbins+1));
			xedges = GetEdges(xdata, n, nbins);
			yedges = GetEdges(ydata, n, nbins);
			Histogram_And_Displacement_2D_steps(hist,hist_before,xedges,yedges,xdata,ydata,n,nbins,steps);
			count = 1;
		}
		void accumulate(Datatype* xdata, Datatype* ydata)
		{
			Histogram_And_Displacement_2D_steps(hist,hist_before,xedges,yedges,xdata,ydata,n,nbins,steps);
			count += 1;
		}
		uint64_t* getHistogram(){return hist;}
		int getCount(){return count;}
		std::tuple<Datatype*,Datatype*> getEdges(){return std::make_tuple(xedges,yedges);}
		int getNbins(){return nbins;}
};

template<class DataType>
void histogram_vectorial_average(int nbins, DataType* hist, DataType* out, int row, int col)
{
	for(int i=0;i<nbins;i++)
	{
		for(int j=0;j<nbins;j++)
		{
			out[0] += hist[i*nbins+j]*(i-row);
			out[1] += hist[i*nbins+j]*(j-col);
		}
	}
}

template<class DataType>
void histogram_nth_order_derivative(int nbins, DataType* data_after, DataType* data_before, DataType dt, int n, int m, DataType* out)
{
	int size = nbins*nbins*nbins*nbins;
	DataType in[2*m+1];
	in[m] = 0;
	for(int i=0;i<nbins;i++)
	{
		for(int j=0;j<nbins;j++)
		{
			for(int k=0;k<nbins;k++)
			{
				for(int l=0;l<nbins;l++)
				{
					for(int s=0;s<m;s++)
					{
						in[m+1+s] = data_after[s*size+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l];
						in[m-1-s] = data_before[s*size+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l];
					}
					nth_order_gradient<DataType>((2*m)+1,in,dt,out+i*nbins*nbins*nbins+j*nbins*nbins+k*nbins+l,n,m);
				}
			}
		}
	}
}
