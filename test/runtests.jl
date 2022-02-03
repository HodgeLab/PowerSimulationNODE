using Test
using Revise
import DataStructures
using Logging
import PowerSystems
using PowerSimulationNODE
using ForwardDiff    #Needed for automated specification of AD through GalacticOptim interface: https://galacticoptim.sciml.ai/stable/API/optimization_function/  
using GalacticOptim  #Needed for automated specification of AD through GalacticOptim interface: https://galacticoptim.sciml.ai/stable/API/optimization_function/ 
using Serialization
using Plots
using PrettyTables

test_file_dir = isempty(dirname(@__FILE__)) ? "test" : dirname(@__FILE__)
const TEST_FILES_DIR = test_file_dir
const PSY = PowerSystems

include("test_generate_and_train.jl")
include("test_hpc.jl")
include("test_serialize.jl")
include("test_prettytable.jl")
