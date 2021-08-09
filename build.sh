#!/bin/bash

make -C FFT/
make -C Correlations/
make -C Math/

mv FFT/obj/libfft.* SignalProcessing/fft/.
mv Math/obj/libmath.* SignalProcessing/math/.
mv Correlations/obj/libcorrelations.* SignalProcessing/correlations/.
