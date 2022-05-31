#!/bin/bash

make -C FFT/
make -C Correlations/
make -C Math/
make -C Histograms/

mv FFT/obj/libfft.* SignalProcessing/fft/.
mv Math/obj/libmath.* SignalProcessing/math/.
mv Correlations/obj/libcorrelations.* SignalProcessing/correlations/.
mv Histograms/obj/libhistograms.* SignalProcessing/histograms/.

while [ "$1" != "" ]; do
		case $1 in
    --enable_cuda)
        make -C FFT_CUDA/
		mv FFT_CUDA/obj/libfftcuda.* SignalProcessing/fft/.
        ;;
	esac
	shift
done

