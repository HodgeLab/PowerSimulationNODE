using Test
using Revise 
using Logging
using PowerSystems
using PowerSimulationsDynamics
using PowerSimulationNODE
using Mustache


test_file_dir = isempty(dirname(@__FILE__)) ? "test" : dirname(@__FILE__)
const TEST_FILES_DIR = test_file_dir
const PSY = PowerSystems


include("test_generate_train_files.jl")
include("test_train.jl")
include("test_hpc.jl")


