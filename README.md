# alpaka_reduction_benchmark
This is a benchmark for a reduction in [alpaka](https://github.com/ComputationalRadiationPhysics/alpaka).

## Requirements

* alpaka 0.3.2
* Boost 1.62+
* a compatible compiler (see [supported compilers](https://github.com/ComputationalRadiationPhysics/alpaka/blob/develop/README.md#supported-compilers) )
* CUDA 8.0+ (optional)

## Build

If CUDA is used, a card with compute capability 3.0 or higher is required. For optimal performance, set ```CMAKE_BUILD_TYPE``` to ```Release```. 

```
mkdir build && cd build
ccmake ..
make 
```

## Usage

```./benchmark```

or

```./benchmark [NUMBER_OFELEMENTS]```

or

``` ./benchmark [START_NUMBER_OF_ELEMENTS] [END_NUMBER_OF_ELEMENTS]```

## License

This project is licensed under GPL v3. For more information see [LICENSE](./LICENSE). 
