# Quique's gpp compilation with cmake

You can compile for debug and for release. Release uses `-O3` and should be used to measure times.

To separate source code from binaries, you can create a debug/release directories. For example:

1. Compile in debug mode

    .../Gpp_prueba$mkdir build-debug && cd build-debug
    .../Gpp_prueba$cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/opt/intel/onetbb ../
    .../Gpp_prueba$make

1. Compile in release mode

    .../Gpp_prueba$mkdir build-release && cd build-release
    .../Gpp_prueba$cmake -DCMAKE_BUILD_TYPE=Release  ../
    .../Gpp_prueba$make
