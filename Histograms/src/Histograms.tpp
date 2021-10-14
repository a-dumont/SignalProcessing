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
		if( (xdata[i]-xmax)*(xdata[i]-xmin)*(ydata[i]-ymax)*(ydata[i]-ymin) >= 0 )
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
		if( (xdata[i]-xmax)*(xdata[i]-xmin)*(ydata[i]-ymax)*(ydata[i]-ymin) >= 0 )
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
void Histogram_And_Displacement_2D(uint32_t* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
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
		if( (xdata[i]-xmax)*(xdata[i]-xmin)*(ydata[i]-ymax)*(ydata[i]-ymin) >= 0 )
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
			hist[nbins*nbins+ybin*nbins*nbins*nbins+xbin*nbins*nbins+nbins*xbin2+ybin2] += 1;
		}
	}
	if( (xdata[n]-xmax)*(xdata[n]-xmin)*(ydata[n]-ymax)*(ydata[n]-ymin) >= 0 )
	{	
		int xbin = (int)((xdata[n]-xmin)*xstep_inv);
		int ybin = (int)((ydata[n]-ymin)*ystep_inv);
		xbin = std::clamp(xbin,0,nbins-1);
		ybin = std::clamp(ybin,0,nbins-1);
		hist[ybin+nbins*xbin] += 1;
	}
}
