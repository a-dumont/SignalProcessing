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
void Histogram(int* hist,Datatype* edges, Datatype* data, int n, int nbins)
{	
	for(int i=0;i<nbins;i++)
	{
		hist[i] = 0;
	}
	Datatype step = edges[1]-edges[0];
	Datatype min = edges[0];	
	for(int i=0;i<n;i++)
	{
		int bin = std::floor((data[i]-min)/step);
		hist[bin] += 1;
	}
}

