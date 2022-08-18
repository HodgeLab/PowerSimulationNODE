# PowerSimulationNODE Documentation

## Background

This package contains the code to train a steady state neural ODE surrogate. A key dependency is `PowerSimulationsDynamicsSurrogates.jl` which contains the code for generating the training data and integrating the trained surrogate into `PowerSystems.jl` and `PowerSimulationsDynamics.jl`. Additionally, `Flux.jl` is used for training the models which make up the surrogate.  

## Getting Started

`TrainParams` is a struct which contains all of the user controlled parameters. `TrainParams` contains parameters related to the system of interest, the generation of training, validation, and test sets, the structure of the surrogate, and the training process itself. Having all of the parameters in one struct makes it easy to directly compare the impact of any change on the surrogate performance. After defining the parameters, there are high level functions for executing the steps of the training:

```julia 
p = TrainParams()   #default parameters 
build_subsystems(p)
generate_train_data(p)
generate_validation_data(p)
generate_test_data(p)
status, θ = train(p)
```

The various parameters are detailed below:
```@docs
TrainParams
```

## HPC 

Training surrogates can be computationally intensive, and often involves hyper-parameter tuning to identify parameters which give the best performance. Therefore, it is desireable to run many training runs in parallel with different sets of parameters. The repository includes code to run sets of trainings on a high performance computing cluster. 

The basic idea is to create two batch files: one for building subsystems and generating data, and a second for executing the actual training. By separating these two steps, we can parallelize across each individually for maximum efficiency. 
```