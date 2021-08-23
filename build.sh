#!/bin/bash

make -C FFT/
make -C Correlations/
make -C Math/
make -C Histograms/

mv FFT/obj/libfft.* SignalProcessing/fft/.
mv Math/obj/libmath.* SignalProcessing/math/.
mv Correlations/obj/libcorrelations.* SignalProcessing/correlations/.
mv Histograms/obj/libhistograms.* SignalProcessing/histograms/.
