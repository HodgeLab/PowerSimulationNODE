#Test development: https://github.com/JuliaLang/Pkg.jl/issues/1973
using Test
using Revise
#using PowerFlows
using PowerSimulationNODE
using PowerSystems
using PowerSimulationsDynamics
using Plots

using Logging
import PowerSystems
using Serialization

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
    include("test_build_param_lists.jl")
end
flush(logger)
close(logger)
