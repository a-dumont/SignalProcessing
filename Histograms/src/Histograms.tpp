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
	for(int i=0;i<nbins;i++)
	{
		hist[i] = 0;
	}
	Datatype step = edges[1]-edges[0];
	Datatype min = edges[0];
	Datatype max = edges[nbins];	
	for(int i=0;i<n;i++)
	{
		if(min <= data[i] && data[i] < max)
		{
			int bin = std::floor((data[i]-min)/step);
			hist[bin] += 1;
		}
		else if (data[i] == max)
		{
			hist[nbins-1] += 1;
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
	Datatype step = edges[1]-edges[0];
	Datatype min = edges[0];
	Datatype max = edges[nbins];
	Datatype norm = 1/step/n;
	for(int i=0;i<n;i++)
	{
		if(min <= data[i] && data[i] < max)
		{
			int bin = std::floor((data[i]-min)/step);
			hist[bin] += norm;
		}
		else if (data[i] == max)
		{
			hist[nbins-1] += norm;
		}

	}
}

template <class Datatype>
void Histogram_2D(long* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
{	
	for(int i=0;i<(nbins*nbins);i++)
	{
		hist[i] = 0;
	}
	Datatype xstep = xedges[1]-xedges[0];
	Datatype ystep = yedges[1]-yedges[0];
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];
	for(int i=0;i<n;i++)
	{
		if(xmin <= xdata[i] && xdata[i] < xmax)
		{	
			if(ymin <= ydata[i] && ydata[i] < ymax)
			{
				int xbin = std::floor((xdata[i]-xmin)/xstep);
				int ybin = std::floor((ydata[i]-ymin)/ystep);
				hist[ybin+(nbins)*xbin] += 1;
			}
			else if( ydata[i] == ymax )
			{
				int xbin = std::floor((xdata[i]-xmin)/xstep);
				int ybin = nbins-1;
				hist[ybin+(nbins)*xbin] += 1;
			}
		}
		else if (xdata[i] == xmax)
		{
			if(ymin <= ydata[i] && ydata[i] < ymax)
			{
				int xbin = nbins-1;
				int ybin = std::floor((ydata[i]-ymin)/ystep);
				hist[ybin+(nbins)*xbin] += 1;
			}
			else if( ydata[i] == ymax )
			{
				int xbin = nbins-1;
				int ybin = nbins-1;
				hist[ybin+(nbins)*xbin] += 1;
			}
		}
	}
}

template <class Datatype>
void Histogram_2D_Density(double* hist, Datatype* xedges, Datatype* yedges, Datatype* xdata, Datatype* ydata, int n, int nbins)
{	
	for(int i=0;i<(nbins*nbins);i++)
	{
		hist[i] = 0;
	}
	Datatype xstep = xedges[1]-xedges[0];
	Datatype ystep = yedges[1]-yedges[0];
	Datatype xmin = xedges[0];
	Datatype ymin = yedges[0];
	Datatype xmax = xedges[nbins];
	Datatype ymax = yedges[nbins];
	Datatype norm = 1/(xstep*ystep)/n;
	for(int i=0;i<n;i++)
	{
		if(xmin <= xdata[i] && xdata[i] < xmax)
		{	
			if(ymin <= ydata[i] && ydata[i] < ymax)
			{
				int xbin = std::floor((xdata[i]-xmin)/xstep);
				int ybin = std::floor((ydata[i]-ymin)/ystep);
				hist[ybin+(nbins)*xbin] += norm;
			}
			else if( ydata[i] == ymax )
			{
				int xbin = std::floor((xdata[i]-xmin)/xstep);
				int ybin = nbins-1;
				hist[ybin+(nbins)*xbin] += norm;
			}
		}
		else if (xdata[i] == xmax)
		{
			if(ymin <= ydata[i] && ydata[i] < ymax)
			{
				int xbin = nbins-1;
				int ybin = std::floor((ydata[i]-ymin)/ystep);
				hist[ybin+(nbins)*xbin] += norm;
			}
			else if( ydata[i] == ymax )
			{
				int xbin = nbins-1;
				int ybin = nbins-1;
				hist[ybin+(nbins)*xbin] += norm;
			}
		}
	}
}
