module PowerSimulationNODE

export train
export NODETrainParams
export serialize
export node_load_system
export SummitHPCTrain
export SavioHPCTrain
export generate_train_files
export run_parallel_train
export fault_data_generator
export build_pvs
export label_area!
export check_single_connecting_line_condition
export remove_area
export build_train_system
export generate_train_data
export NODETrainDataParams
export visualize_summary
export generate_summary
export visualize_training
export animate_training
export print_train_parameter_overview

import Arrow
import DataFrames
import DataStructures
import DiffEqBase
import DiffEqSensitivity
import FFTW
import Flux
import Flux.Losses: mae, mse
import GalacticOptim
import IterTools
import JSON3
import Logging
import Mustache
import NLsolve
import Optim
import OrdinaryDiffEq
import Plots 
import PowerSimulationsDynamics
import PowerSystems
import Random
import Serialization
import StatsBase
import SciMLBase
import StructTypes
import YAML
import PrettyTables
import Zygote
const PSY = PowerSystems
const PSID = PowerSimulationsDynamics


#TODO Split up code and use Requires strategically to improve load times (especially the instantiate functions)
include(joinpath("train", "Theta.jl"))
include(joinpath("train", "surrogate_models.jl"))
include(joinpath("train", "NODETrainParams.jl"))
include("constants.jl")
include(joinpath("generate_data", "fault_pvs.jl"))
include(joinpath("train", "HPCTrain.jl"))
include(joinpath("generate_data", "NODETrainInputs.jl"))
include(joinpath("train", "train.jl"))
include(joinpath("train", "instantiate.jl"))
include(joinpath("train", "multiple_shoot.jl"))
include(joinpath("visualize", "visualize.jl"))
include("utils.jl")
end
