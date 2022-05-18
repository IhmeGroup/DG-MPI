# Build scripts
The following set of scripts can be used for the build process. For each of these it is assumed that you have your own build of cmake since Kokkos requires a cmake version newer than the one provided on the Yellowstone cluster.

## For a Yellowstone build (yellowstone-cpu.sh or yellowstone-gpu.sh):
This file should be called from the 'DG-MPI/src/others' folder.

NOTE: Make sure you follow directions to install cmake version 3.22 on the
github page (see below) and then run this command as follows:

`cmake=<PATH_TO_CMAKE>/cmake-3.22.2/bin/cmake ./yellowstone-cpu.sh`

## IMPORTANT! 
For Yellowstone Builds: Kokkos requires CMake >= 3.16. Yellowstone has version 3.15.4. Therefore, we need to install a local build of CMake. Use the following commands to accomplish this:

         - Download the Unix/Linux source cmake file (I used version 3.22.2) `https://cmake.org/download/`
         - Send this to yellowstone using `rsync` or whatever your favorite file transfer tool is.
         - Once you have the `*tar.gz` file where you want the CMake build to happen use these commands:
         ```
         tar -zxvf cmake-3.22.2.tar.gz
         cd cmake-3.22.2
         mkdir install
         cmake -DCMAKE_USE_OPENSSL=OFF -DCMAKE_INSTALL_PREFIX=<path_to_your_cmake>/install .
         make
         make install
         ```
         - Then when you go to use CMake with Kokkos you can either give the direct path to the new cmake command when executing the cmake command for Kokkos above or change your default CMake to the newly installed version (by changing the paths / unloading the module for CMake).

## Some tips:
On occasion it was noted that the `kokkos-kernels` build fails with an error of `Permission Denied` specifically in the GPU build of the code. Ways of dealing with this include trying to build again (sometimes it just works) or I have had success by doing the following:
- `cd ../../build_gpu/externals/kokkos-kernels`
- `make install`
For some reason this tends to work, although it is slow. This should be revisited at a future date to figure out why the build is not consistent.
