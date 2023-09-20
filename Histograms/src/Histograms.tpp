#include <stdexcept>

////////////////////////////////////////////////////////////////
//  _     _   _   _ _     _                                   //
// / | __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//                                 |___/                      //
////////////////////////////////////////////////////////////////

template <class DataType>
void histogram(uint32_t* hist, DataType* data, DataType* edges, uint64_t N, uint64_t nbins)
{	
	double step_inv = 1.0/(edges[1]-edges[0]);
	DataType min = edges[0];
	DataType max = edges[nbins];
	uint64_t zero = 0;
	
	#pragma omp parallel for reduction(+:hist[:nbins])
	for(uint64_t i=0;i<N;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			uint64_t bin = (uint64_t) ((data[i]-min)*step_inv);
			bin = std::clamp(bin,zero,nbins-1);
			hist[bin] += 1;
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

////////////////////////////////////////////////////////////////
//  _     _   _     _     _                                   //
// / | __| | | |__ (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
// | |/ _` | | '_ \| / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
// | | (_| | | | | | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_|\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//	                               |___/                      //
//			      _                _ _                        //
//             __| | ___ _ __  ___(_) |_ _   _                //
//			  / _` |/ _ \ '_ \/ __| | __| | | |               //
//			 | (_| |  __/ | | \__ \ | |_| |_| |               //
//			  \__,_|\___|_| |_|___/_|\__|\__, |               //
//			                             |___/                //
////////////////////////////////////////////////////////////////

template <class DataTypeIn, class DataTypeOut>
void histogram_density(DataTypeOut* hist, DataTypeIn* data, DataTypeIn* edges, 
				uint64_t N, uint64_t nbins, bool density)
{	
	DataTypeOut step_inv = (DataTypeOut) (1.0/(edges[1]-edges[0]));
	DataTypeOut norm;
	if(density == true){norm = (DataTypeOut) (step_inv/N);}
	else{norm = (DataTypeOut) (1.0);}

	DataTypeIn min = edges[0];
	DataTypeIn max = edges[nbins];
	uint64_t zero = 0;
	
	#pragma omp parallel for reduction(+:hist[:nbins])
	for(uint64_t i=0;i<N;i++)
	{
		if((data[i]-max)*(data[i]-min) <= 0)
		{
			uint64_t bin = (uint64_t) ((data[i]-min)*step_inv);
			bin = std::clamp(bin,zero,nbins-1);
			hist[bin] += norm;
		}
	}
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
///////////////////////////////////////////////////////////////////

template <class DataType>
void histogram2D(uint32_t* hist, DataType* xedges, DataType* yedges,
				DataType* xdata, DataType* ydata, uint64_t N, uint64_t nbins)
{	
	DataType xstep_inv = 1.0/(xedges[1]-xedges[0]);
	DataType ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	uint64_t zero = 0;
	
	for(uint64_t i=0;i<N;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				uint64_t xbin = (uint64_t) ((xdata[i]-xmin)*xstep_inv);
				uint64_t ybin = (uint64_t) ((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,zero,nbins-1);
				ybin = std::clamp(ybin,zero,nbins-1);
				hist[ybin+nbins*xbin] += 1;
		}
	}
}

template<class DataType>
void digitizer_histogram2D(uint32_t* hist, DataType* data_x, DataType* data_y, uint64_t N)
{
	uint64_t n = 8*sizeof(DataType);
	uint64_t remaining = N/8;
	#pragma omp parallel for reduction(+:hist[:1<<(2*n)])
	for(uint64_t i=0; i<(N-7); i+=8)
	{
		hist[data_y[ i ] ^ (data_x[ i ]<<n)] += 1;
		hist[data_y[i+1] ^ (data_x[i+1]<<n)] += 1;
		hist[data_y[i+2] ^ (data_x[i+2]<<n)] += 1;
		hist[data_y[i+3] ^ (data_x[i+3]<<n)] += 1;
		hist[data_y[i+4] ^ (data_x[i+4]<<n)] += 1;
		hist[data_y[i+5] ^ (data_x[i+5]<<n)] += 1;
		hist[data_y[i+6] ^ (data_x[i+6]<<n)] += 1;
		hist[data_y[i+7] ^ (data_x[i+7]<<n)] += 1;
	}
	for(uint64_t j=(8*remaining); j<(N); j++)
	{
		hist[data_y[j] ^ (data_x[j]<<n)] += 1;
	}
}

template<class DataType>
void digitizer_histogram2D_subbyte(uint32_t* hist, DataType* data_x, 
				DataType* data_y, uint64_t N, uint64_t nbits)
{
	uint64_t size = 1<<nbits;
	uint8_t shift = sizeof(DataType)*8-nbits;
	uint64_t remaining = N/8;
	#pragma omp parallel for reduction(+:hist[:size<<nbits])
	for(uint64_t i=0; i<(N-7); i+=8)
	{
		hist[(data_y[ i ]>>shift) ^ ((data_x[ i ]>>shift)<<nbits)] += 1;
		hist[(data_y[i+1]>>shift) ^ ((data_x[i+1]>>shift)<<nbits)] += 1;
		hist[(data_y[i+2]>>shift) ^ ((data_x[i+2]>>shift)<<nbits)] += 1;
		hist[(data_y[i+3]>>shift) ^ ((data_x[i+3]>>shift)<<nbits)] += 1;
		hist[(data_y[i+4]>>shift) ^ ((data_x[i+4]>>shift)<<nbits)] += 1;
		hist[(data_y[i+5]>>shift) ^ ((data_x[i+5]>>shift)<<nbits)] += 1;
		hist[(data_y[i+6]>>shift) ^ ((data_x[i+6]>>shift)<<nbits)] += 1;
		hist[(data_y[i+7]>>shift) ^ ((data_x[i+7]>>shift)<<nbits)] += 1;
	}
	for(uint64_t j=(8*remaining); j<(N); j++)
	{
		hist[(data_y[j]>>shift) ^ ((data_x[j]>>shift)<<nbits)] += 1;
	}
}

template<class DataType>
void digitizer_histogram2D_10bits(uint16_t* hist, DataType* data_x, DataType* data_y, uint64_t N)
{
	uint64_t remaining = N/8;
	for(uint64_t i=0; i<(N-7); i+=8)
	{
		hist[data_y[ i ] ^ (data_x[ i ]<<10)] += 1;
		hist[data_y[i+1] ^ (data_x[i+1]<<10)] += 1;
		hist[data_y[i+2] ^ (data_x[i+2]<<10)] += 1;
		hist[data_y[i+3] ^ (data_x[i+3]<<10)] += 1;
		hist[data_y[i+4] ^ (data_x[i+4]<<10)] += 1;
		hist[data_y[i+5] ^ (data_x[i+5]<<10)] += 1;
		hist[data_y[i+6] ^ (data_x[i+6]<<10)] += 1;
		hist[data_y[i+7] ^ (data_x[i+7]<<10)] += 1;
	}
	for(uint64_t j=(8*remaining); j<(N); j++)
	{
		hist[data_y[j]^ (data_x[j]<<10)] += 1;
	}
}

///////////////////////////////////////////////////////////////////
//  ____     _   _   _ _     _                                   //
// |___ \ __| | | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___   //
//   __) / _` | | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \  //
//  / __/ (_| | |  _  | \__ \ || (_) | (_| | | | (_| | | | | | | //
// |_____\__,_| |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_| //
//					                  |___/                      //
//			      _                _ _                           //
//             __| | ___ _ __  ___(_) |_ _   _                   //
//			  / _` |/ _ \ '_ \/ __| | __| | | |                  //
//			 | (_| |  __/ | | \__ \ | |_| |_| |                  //
//			  \__,_|\___|_| |_|___/_|\__|\__, |                  //
//			                             |___/                   //
///////////////////////////////////////////////////////////////////

template <class DataTypeIn, class DataTypeOut>
void histogram2D_density(DataTypeOut* hist, DataTypeIn* xedges, DataTypeIn* yedges,
				DataTypeIn* xdata, DataTypeIn* ydata, uint64_t N, uint64_t nbins, bool density)
{	
	DataTypeOut xstep_inv = 1.0/(xedges[1]-xedges[0]);
	DataTypeOut ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataTypeIn xmin = xedges[0];
	DataTypeIn ymin = yedges[0];
	DataTypeIn xmax = xedges[nbins];
	DataTypeIn ymax = yedges[nbins];
	DataTypeOut norm = xstep_inv*ystep_inv/N;
	if(density==true){norm = xstep_inv*ystep_inv/N;}
	else{norm = 1.0;}
	uint64_t zero = 0;
	
	for(uint64_t i=0;i<N;i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
				uint64_t xbin = (uint64_t) ((xdata[i]-xmin)*xstep_inv);
				uint64_t ybin = (uint64_t) ((ydata[i]-ymin)*ystep_inv);
				xbin = std::clamp(xbin,zero,nbins-1);
				ybin = std::clamp(ybin,zero,nbins-1);
				hist[ybin+nbins*xbin] += norm;
		}
	}
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
void histogram2D_step(uint32_t* hist, DataType* xedges, DataType* yedges, 
				DataType* xdata, DataType* ydata, uint64_t N, uint64_t nbins)
{	
	double xstep_inv = 1.0/(xedges[1]-xedges[0]);
	double ystep_inv = 1.0/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	uint64_t zero = (uint64_t) 0;

	#pragma omp parallel for
	for(uint64_t i=0;i<(N-1);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			uint64_t xbin = (uint64_t) ((xdata[i]-xmin)*xstep_inv);
			uint64_t ybin = (uint64_t) ((ydata[i]-ymin)*ystep_inv);
			uint64_t xbin2 = (uint64_t) ((xdata[i+1]-xmin)*xstep_inv);
			uint64_t ybin2 = (uint64_t) ((ydata[i+1]-ymin)*ystep_inv);
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
}

template<class DataType>
void histogram2D_steps(uint32_t* hist_after, uint32_t* hist_before, DataType* xedges, DataType* yedges, DataType* xdata, DataType* ydata, uint64_t n, uint64_t nbins, uint64_t steps)
{	
	DataType xstep_inv = 1/(xedges[1]-xedges[0]);
	DataType ystep_inv = 1/(yedges[1]-yedges[0]);
	DataType xmin = xedges[0];
	DataType ymin = yedges[0];
	DataType xmax = xedges[nbins];
	DataType ymax = yedges[nbins];
	uint64_t xbin,ybin,xbin2,ybin2,xbin3,ybin3;
	uint64_t zero = 0;

	for(uint64_t i=steps;i<(n-steps);i++)
	{
		if( ((xdata[i]-xmax)*(xdata[i]-xmin) <= 0) && ((ydata[i]-ymax)*(ydata[i]-ymin) <= 0) )
		{	
			xbin = std::clamp((uint64_t) ((xdata[i]-xmin)*xstep_inv),zero,nbins-1);
			ybin = std::clamp((uint64_t) ((ydata[i]-ymin)*ystep_inv),zero,nbins-1);
			hist_after[ybin+nbins*xbin] += 1;

			for(uint64_t j=1;j<steps+1;j++)
			{
				xbin2 = std::clamp((uint64_t) ((xdata[i+j]-xmin)*xstep_inv),zero,nbins-1);
				ybin2 = std::clamp((uint64_t) ((ydata[i+j]-ymin)*ystep_inv),zero,nbins-1);
				xbin3 = std::clamp((uint64_t) ((xdata[i-j]-xmin)*xstep_inv),zero,nbins-1);
				ybin3 = std::clamp((uint64_t) ((ydata[i-j]-ymin)*ystep_inv),zero,nbins-1);
				
				hist_after[nbins*nbins+(xbin*nbins*nbins*nbins+ybin*nbins*nbins)
						+nbins*xbin2+ybin2+(j-1)*nbins*nbins*nbins*nbins] += 1;
				
				hist_before[(xbin*nbins*nbins*nbins+ybin*nbins*nbins)+nbins*xbin3+ybin3
						+(j-1)*nbins*nbins*nbins*nbins] += 1;
			}
		}
	}
}

template<class DataType>
void digitizer_histogram2D_step(uint32_t* hist, DataType* data_x, 
				DataType* data_y, uint64_t N, uint8_t nbits)
{
	uint8_t shift = sizeof(DataType)*8-nbits;
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

template<class DataType>
void digitizer_histogram2D_step_10bit(uint32_t* hist, DataType* data_x, DataType* data_y, uint64_t N)
{
	uint64_t bin_x, bin_y, bin_x2, bin_y2;
	uint32_t* hist2 = hist+(1<<20);

	for(uint64_t i=0; i<(N-1); i++)
	{
		bin_x = data_x[i];
		bin_y = data_y[i];
		bin_x2 = data_x[i+1];
		bin_y2 = data_y[i+1];
	
		hist[bin_y ^ (bin_x<<10)] += 1;	
		hist2[bin_y2 ^ (bin_x<<30) ^ (bin_y<<20) ^ (bin_x2<<10)] += 1;
	}
}

//////////////////////////////////////////////////
//   __                  _   _                  //
//  / _|_   _ _ __   ___| |_(_) ___  _ __  ___  //
// | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __| //
// |  _| |_| | | | | (__| |_| | (_) | | | \__ \ //
// |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ //
//                                              //
//////////////////////////////////////////////////

template <class DataType>
void get_edges(DataType* data, uint64_t N, uint64_t nbins, DataType* edges)
{
	std::pair<DataType*,DataType*> minmax = std::minmax_element(data,data+N);
	DataType min = *minmax.first;
	DataType max = *minmax.second;
	DataType step = (DataType) (max-min)/nbins;
	edges[0] = min;
	edges[nbins] = max;
	for(uint64_t i=1;i<(nbins);i++)
	{
		edges[i] = edges[i-1]+step;
	}
}

template <class DataType>
uint64_t find_first_in_bin(DataType* data, DataType* edges, uint64_t N, uint64_t bin)
{
	for(uint64_t i=0;i<N;i++)
	{
		if(data[i]>=edges[bin] && data[i] <= edges[bin+1])
		{
			return i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template <class DataType>
uint64_t find_first_in_bin2D(DataType* xdata, DataType* ydata, DataType* xedges, DataType* yedges, 
				uint64_t N, uint64_t xbin, uint64_t ybin)
{
	for(uint64_t i=0;i<N;i++)
	{
		if
		(
			xdata[i]>=xedges[xbin]&&
			xdata[i]<=xedges[xbin+1]&&
			ydata[i]>=yedges[ybin]&&
			ydata[i]<=yedges[ybin+1]
		)
		{
			return i;
		}
	}
	throw std::runtime_error("No value in range.");
}

template<class DataType>
void histogram_vectorial_average(uint64_t nbins, 
				DataType* hist, DataType* out, uint64_t row, uint64_t col)
{
	double norm;
	//#pragma omp parallel for reduction(+:out[:2])
	for(uint64_t i=0;i<nbins;i++)
	{
		for(uint64_t j=0;j<nbins;j++)
		{
			if(i!=row && j!=col)
			{
				norm = sqrt((1.0*i-1.0*row)*(1.0*i-1.0*row)+(1.0*j-1.0*col)*(1.0*j-1.0*col));
				out[0] += hist[i*nbins+j]*(1.0*i-1.0*row)/norm;
				out[1] += hist[i*nbins+j]*(1.0*j-1.0*col)/norm;
			}
		}
	}
	out[0] /= 1.0*nbins*nbins;
	out[1] /= 1.0*nbins*nbins;
}

template<class DataType>
void histogram_nth_order_derivative(uint64_t nbins, DataType* data_after, DataType* data_before, 
				DataType dt, uint64_t m, uint64_t n, DataType* out, DataType* coeff)
{
	uint64_t size = nbins*nbins*nbins*nbins;
	DataType in[2*n+1];
	in[n] = 0;
	//#pragma omp parallel for
	for(uint64_t i=0;i<nbins;i++)
	{
		for(uint64_t j=0;j<nbins;j++)
		{
			for(uint64_t k=0;k<nbins;k++)
			{
				for(uint64_t l=0;l<nbins;l++)
				{
					for(uint64_t s=0;s<n;s++)
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
void detailed_balance(uint64_t bins, DataType* p_density, DataType* gamma, DataType* out)
{
	#pragma omp parallel for
	for(uint64_t i=0;i<bins;i++)
	{
		for(uint64_t j=0;j<bins;j++)
		{
			for(uint64_t k=0;k<bins;k++)
			{
				for(uint64_t l=0;l<bins;l++)
				{
					out[i*bins*bins*bins+j*bins*bins+k*bins+l] = 
					p_density[i*bins+j]*gamma[i*bins*bins*bins+j*bins*bins+k*bins+l]-
					p_density[k*bins+l]*gamma[k*bins*bins*bins+l*bins*bins+i*bins+j];
				}
			}
		}
	}
}

///////////////////////////////////////
//       _                           //
//   ___| | __ _ ___ ___  ___  ___   //
//  / __| |/ _` / __/ __|/ _ \/ __|  //
// | (__| | (_| \__ \__ \  __/\__ \  //
//  \___|_|\__,_|___/___/\___||___/  //
//                                   //
///////////////////////////////////////                                      

class Histogram2D 
{
	protected:
		uint32_t* hist;
		uint64_t nbins, count, size;
		double *xedges, *yedges;
	public:
		Histogram2D(uint64_t nbins_in)
		{
			count = 0;
			nbins = nbins_in;
			size = nbins*nbins;

			hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
			std::memset(hist,0,sizeof(uint32_t)*size);
			
			xedges = (double*) malloc(sizeof(double)*(nbins+1));
			yedges = (double*) malloc(sizeof(double)*(nbins+1));
		}
		~Histogram2D(){free(hist);free(xedges);free(yedges);}
		
		template <class DataType>
		void initialize(DataType* xdata, DataType* ydata, uint64_t N)
		{
			resetHistogram();
			resetEdges();

			get_edges(xdata, N, nbins, (DataType*) xedges);
			get_edges(ydata, N, nbins, (DataType*) yedges);
			
			histogram2D<DataType>(hist,(DataType*) xedges,(DataType*) yedges,xdata,ydata,N,nbins);
			count += 1;
		}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			histogram2D<DataType>(hist,(DataType*) xedges,(DataType*) yedges,xdata,ydata,N,nbins);
			count += 1;
		}
		
		template <class DataType>
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		
		void resetHistogram(){std::memset(hist,0,size*sizeof(uint32_t));count=0;}
		
		void resetEdges()
		{
			std::memset(xedges,0,(nbins+1)*sizeof(double));
			std::memset(yedges,0,(nbins+1)*sizeof(double));
		}
		
		uint32_t* getHistogram(){return hist;}
		uint64_t getCount(){return count;}
		uint64_t getNbins(){return nbins;}
};

class Histogram2D_Density 
{
	protected:
		double* hist;
		uint64_t nbins, count, size;
		double *xedges, *yedges;
	public:
		Histogram2D_Density(uint64_t nbins_in)
		{
			count = 0;
			nbins = nbins_in;
			size = nbins*nbins;

			hist = (double*) malloc(sizeof(double)*size);
			std::memset(hist,0,sizeof(double)*size);
			
			xedges = (double*) malloc(sizeof(double)*(nbins+1));
			yedges = (double*) malloc(sizeof(double)*(nbins+1));
		}
		~Histogram2D_Density(){free(hist);free(xedges);free(yedges);}
		
		template <class DataType>
		void initialize(DataType* xdata, DataType* ydata, uint64_t N)
		{
			resetHistogram();
			resetEdges();

			get_edges(xdata, N, nbins, (DataType*) xedges);
			get_edges(ydata, N, nbins, (DataType*) yedges);
			
			histogram2D_density<DataType>(hist,(DataType*) xedges,(DataType*) yedges,
							xdata,ydata,N,nbins,1);
			count += 1;
		}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			histogram2D_density<DataType>(hist,(DataType*) xedges,(DataType*) yedges,
							xdata,ydata,N,nbins,1);
			count += 1;
		}
		
		template <class DataType>
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		
		void resetHistogram(){std::memset(hist,0,size*sizeof(double));count=0;}
		
		void resetEdges()
		{
			std::memset(xedges,0,(nbins+1)*sizeof(double));
			std::memset(yedges,0,(nbins+1)*sizeof(double));
		}
		
		double* getHistogram(){return hist;}
		uint64_t getCount(){return count;}
		uint64_t getNbins(){return nbins;}
};

class Histogram2D_Step 
{
	protected:
		uint32_t* hist;
		uint64_t nbins, count, size;
		double *xedges, *yedges;
	public:
		Histogram2D_Step(uint64_t nbins_in)
		{
			count = 0;
			nbins = nbins_in;
			size = nbins*nbins*(1+nbins*nbins);

			hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
			std::memset(hist,0,sizeof(uint32_t)*size);
			
			xedges = (double*) malloc(sizeof(double)*(nbins+1));
			yedges = (double*) malloc(sizeof(double)*(nbins+1));
		}
		~Histogram2D_Step(){free(hist);free(xedges);free(yedges);}
		
		template <class DataType>
		void initialize(DataType* xdata, DataType* ydata, uint64_t N)
		{
			resetHistogram();
			resetEdges();

			get_edges(xdata, N, nbins, (DataType*) xedges);
			get_edges(ydata, N, nbins, (DataType*) yedges);
			
			histogram2D_step<DataType>(hist,(DataType*) xedges,(DataType*) yedges,
							xdata,ydata,N,nbins);
			count += 1;
		}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			histogram2D_step<DataType>(hist,(DataType*) xedges,(DataType*) yedges,
							xdata,ydata,N,nbins);
			count += 1;
		}
		
		template <class DataType>
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		
		void resetHistogram(){std::memset(hist,0,size*sizeof(uint32_t));count=0;}
		
		void resetEdges()
		{
			std::memset(xedges,0,(nbins+1)*sizeof(double));
			std::memset(yedges,0,(nbins+1)*sizeof(double));
		}
		
		uint32_t* getHistogram(){return hist;}
		uint64_t getCount(){return count;}
		uint64_t getNbins(){return nbins;}
};

class Histogram2D_Steps 
{
	protected:
		uint32_t* hist;
		uint32_t* hist_after;
		uint32_t* hist_before;
		uint64_t nbins, count, size, steps;
		double *xedges, *yedges;
	public:
		Histogram2D_Steps(uint64_t nbins_in, uint64_t steps_in)
		{
			count = 0;
			steps = steps_in;
			nbins = nbins_in;
			size = nbins*nbins*(1+2*steps*nbins*nbins);

			hist = (uint32_t*) malloc(sizeof(uint32_t)*size);
			std::memset(hist,0,sizeof(uint32_t)*size);

			hist_after = hist;
			hist_before = hist+nbins*nbins*(1+steps*nbins*nbins);
			
			xedges = (double*) malloc(sizeof(double)*(nbins+1));
			yedges = (double*) malloc(sizeof(double)*(nbins+1));
		}
		~Histogram2D_Steps(){free(hist);free(xedges);free(yedges);}
		
		template <class DataType>
		void initialize(DataType* xdata, DataType* ydata, uint64_t N)
		{
			resetHistogram();
			resetEdges();

			get_edges(xdata, N, nbins, (DataType*) xedges);
			get_edges(ydata, N, nbins, (DataType*) yedges);
			
			histogram2D_steps<DataType>(hist_after,hist_before,(DataType*) xedges,
							(DataType*) yedges,xdata,ydata,N,nbins,steps);
			count += 1;
		}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			histogram2D_steps<DataType>(hist_after,hist_before,(DataType*) xedges,
							(DataType*) yedges,xdata,ydata,N,nbins,steps);
			count += 1;
		}
		
		template <class DataType>
		std::tuple<DataType*,DataType*> getEdges(){return std::make_tuple(xedges,yedges);}
		
		void resetHistogram(){std::memset(hist,0,size*sizeof(uint32_t));count=0;}
		
		void resetEdges()
		{
			std::memset(xedges,0,(nbins+1)*sizeof(double));
			std::memset(yedges,0,(nbins+1)*sizeof(double));
		}
		
		uint32_t* getHistogram(){return hist;}
		uint64_t getCount(){return count;}
		uint64_t getNbins(){return nbins;}
};

class Digitizer_histogram2D_step
{
	protected:
		uint32_t* hist;
		uint64_t* hist_out;
		bool hist_out_init;
		uint64_t nbits;
		uint64_t size;
		uint64_t total_size;
		uint64_t count;
		uint64_t N_t;
	public:
		Digitizer_histogram2D_step(uint64_t nbits_in)
		{
			if(nbits_in > (uint64_t) 10)
			{throw std::runtime_error("U dumbdumb nbits to large for parallel reduction.");}
			nbits = nbits_in;
			size = 1<<nbits;
			total_size = size*size*(size*size+1);
			count = 0;
            #ifdef _WIN32_WINNT
                uint64_t nbgroups = GetActiveProcessorGroupCount();
                N_t = std::min((uint64_t) 64,omp_get_max_threads()*nbgroups);
            #else
                N_t = omp_get_max_threads();
			#endif
            
			hist = (uint32_t*) malloc(N_t*sizeof(uint32_t)*size*size*(size*size+1));

			std::memset(hist,0,N_t*sizeof(uint32_t)*size*size*(size*size+1));
			hist_out_init = false;
		}
		
		~Digitizer_histogram2D_step(){free(hist);if(hist_out_init == true){free(hist_out);}}

		template <class DataType>
		void accumulate(DataType* xdata, DataType* ydata, uint64_t N)
		{
			N = N/N_t;
			#pragma omp parallel for num_threads(N_t)
			for(uint64_t i=0;i<N_t;i++)
			{
                manage_thread_affinity();
				digitizer_histogram2D_step(hist+i*total_size,xdata+i*N,ydata+i*N,N,nbits);
			}
			count += 1;
		}
		void resetHistogram()
		{
			std::memset(hist,0,N_t*sizeof(uint32_t)*size*size*(size*size+1));
			if(hist_out_init == true)
			{
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(size*size+1));
			}
			count = 0;
		}
		uint64_t* getHistogram()
		{
			if(hist_out_init == false)
			{
				hist_out = (uint64_t*) malloc(sizeof(uint64_t)*size*size*(size*size+1));
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(size*size+1));
				hist_out_init = true;
			}
			else
			{
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(size*size+1));	
			}
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
		uint64_t getSize(){return size;}
		uint64_t getThreads(){return N_t;}
};

class Digitizer_histogram2D_steps
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
		Digitizer_histogram2D_steps(uint64_t nbits_in, uint64_t steps_in)
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
		
		~Digitizer_histogram2D_steps(){free(hist);if(hist_out_init == true){free(hist_out);}}

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
				hist_out = (uint64_t*) malloc(sizeof(uint64_t)*size*size*(2*steps*size*size+1));
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(2*steps*size*size+1));
				hist_out_init = true;
			}
			else
			{
				std::memset(hist_out,0,sizeof(uint64_t)*size*size*(2*steps*size*size+1));	
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
