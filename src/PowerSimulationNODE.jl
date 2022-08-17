module PowerSimulationNODE

export train
export TrainParams
export serialize
export node_load_system
export SummitHPCTrain
export SavioHPCTrain
export generate_train_files
export generate_pvs_data
export run_parallel_train
export label_area!
export generate_surrogate_data
export GenerateDataParams
export visualize_summary
export generate_summary
export visualize_training
export animate_training
export print_train_parameter_overview
export print_high_level_output_overview
export build_params_list!
export create_surrogate_training_system

import Arrow
import Dates
import DataFrames
import DataStructures
import DiffEqBase
import DiffEqSensitivity
import FFTW
import Flux
import Flux.Losses: mae, mse
import GalacticOptim
import GalacticFlux
import IterTools
import JSON3
import Logging
import Mustache
import NLsolve
import Optim
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
import StructTypes
import YAML
import PrettyTables
import Zygote
using LaTeXStrings
const PSY = PowerSystems
const PSID = PowerSimulationsDynamics
const PSIDS = PowerSimulationsDynamicsSurrogates

#TODO Split up code and use Requires strategically to improve load times (especially the instantiate functions)
include(joinpath("train", "SteadyStateNeuralODE.jl"))
include(joinpath("train", "TrainParams.jl"))
include("constants.jl")
include(joinpath("train", "HPCTrain.jl"))
#include(joinpath("generate_data", "data_containers.jl"))
#include(joinpath("generate_data", "generate_data.jl"))
#include(joinpath("generate_data", "build_systems.jl"))
include(joinpath("train", "train.jl"))
include(joinpath("train", "instantiate.jl"))
include(joinpath("visualize", "visualize.jl"))
include("utils.jl")
end
