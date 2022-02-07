# Installing Dependencies

Modules to load:
	```
	module load cuda
	module load mpich
	module load hdf5
	```

Get `metis` by running `./download_dependencies.sh`


## Kokkos

	```
	cd ../..
	cd kokkos-kernels
	mkdir  -p build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install -DCMAKE_CXX_COMPILER=${PWD}/../../kokkos/build/install/bin/nvcc_wrapper -DKokkosKernels_REQUIRE_DEVICES=CUDA -DKokkos_ARCH_KEPLER35=ON ..
	```

## Kokkos-kernels
	```
	cd kokkos-kernels
	mkdir  -p build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install -DCMAKE_CXX_COMPILER=${PWD}/../../kokkos/build/install/bin/nvcc_wrapper -DKokkosKernels_REQUIRE_DEVICES=CUDA ..
	```

## Main source code
	```
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_CXX_COMPILER=${PWD}/../externals/kokkos/build/install/bin/nvcc_wrapper ..