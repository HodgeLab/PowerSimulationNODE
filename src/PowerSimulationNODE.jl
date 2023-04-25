module PowerSimulationNODE

export train
export TrainParams
export serialize
export node_load_system
export node_run_powerflow!
export AlpineHPCTrain
export SavioHPCTrain
export generate_train_files
export copy_data_and_subsystems
export run_parallel_train
export generate_surrogate_data
export GenerateDataParams
export visualize_summary
export generate_summary
export rebase_path!
export visualize_training
export animate_training
export print_train_parameter_overview
export print_high_level_output_overview
export build_grid_search!
export build_random_search!
export build_subsystems
export generate_train_data
export generate_validation_data
export generate_test_data
export evaluate_loss
export visualize_loss
export generate_surrogate_dataset

import Arrow
import Dates
import DataFrames
import DiffEqBase
import DiffEqCallbacks
import Flux
import Flux.Losses: mae, mse
import Optimization
import IterTools
import JSON3
import Logging
import Mustache
import NLsolve
import Optim
import Optimization
import OptimizationOptimisers
import OptimizationOptimJL
import OrdinaryDiffEq
import SteadyStateDiffEq
import Plots
import PowerFlows
import PowerSimulationsDynamics
import PowerSimulationsDynamicsSurrogates
import PowerSystems
import Random
import Serialization
import StatsBase
import SciMLBase
import SciMLSensitivity
import Statistics
import StructTypes
import YAML
import PrettyTables
import Zygote
using LaTeXStrings
const PSY = PowerSystems
const PSID = PowerSimulationsDynamics
const PSIDS = PowerSimulationsDynamicsSurrogates

# Split up code and use Requires strategically to improve load times? (especially the instantiate functions)
include(joinpath("train", "surrogates", "common_control_overloads.jl"))
include(joinpath("train", "surrogates", "solution_structs.jl"))
include(joinpath("train", "surrogates", "SteadyStateNeuralODE.jl"))
include(joinpath("train", "surrogates", "ClassicGen.jl"))
include(joinpath("train", "surrogates", "GFL.jl"))
include(joinpath("train", "surrogates", "GFM.jl"))
include(joinpath("train", "surrogates", "ZIP.jl"))
include(joinpath("train", "surrogates", "MultiDevice.jl"))
include(joinpath("train", "TrainParams.jl"))
include("constants.jl")
include(joinpath("train", "HPCTrain.jl"))
include(joinpath("train", "generate.jl"))
include(joinpath("train", "train.jl"))
include(joinpath("train", "instantiate.jl"))
include(joinpath("visualize", "visualize.jl"))
include("utils.jl")
end
