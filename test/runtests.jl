#Test development: https://github.com/JuliaLang/Pkg.jl/issues/1973
using Test
using Revise
using PowerSimulationNODE   #Order matter for using Revise? Possibly issue with Requires? 

import DataStructures
using Logging
import PowerSystems
using ForwardDiff    #Needed for automated specification of AD through GalacticOptim interface: https://galacticoptim.sciml.ai/stable/API/optimization_function/  
using GalacticOptim  #Needed for automated specification of AD through GalacticOptim interface: https://galacticoptim.sciml.ai/stable/API/optimization_function/ 
using Serialization
using Plots
using PrettyTables

test_file_dir = isempty(dirname(@__FILE__)) ? "test" : dirname(@__FILE__)
const TEST_FILES_DIR = test_file_dir
const PSY = PowerSystems

logger = PSY.configure_logging(;
    console_level = PowerSimulationNODE.NODE_CONSOLE_LEVEL,
    file_level = PowerSimulationNODE.NODE_FILE_LEVEL,
)
with_logger(logger) do
    include("test_generate_train.jl")
    include("test_hpc.jl")
    include("test_serialize.jl")
    include("test_prettytable.jl")
end
flush(logger)
close(logger)
