using Test
using Revise
using Logging
using PowerSystems
using PowerSimulationsDynamics
using PowerSimulationNODE

test_file_dir = isempty(dirname(@__FILE__)) ? "test" : dirname(@__FILE__)
const TEST_FILES_DIR = test_file_dir
const PSY = PowerSystems

include("test_generate_and_train.jl")
include("test_hpc.jl")
