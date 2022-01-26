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
export visualize_training

import Arrow
import DataFrames
import DataStructures
import DiffEqBase
import DiffEqFlux
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
import PowerSimulationsDynamics
import PowerSystems
import Random
import StatsBase
import SciMLBase
import StructTypes
import YAML

const PSY = PowerSystems
const PSID = PowerSimulationsDynamics

using Requires
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include(
        joinpath("visualize", "visualize.jl"),
    )
    @require GalacticOptim = "a75be94c-b780-496d-a8a9-0878b188d577" begin
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include(
            joinpath("train", "instantiate.jl"),
        )
    end
end

#TODO Split up code and use Requires strategically to improve load times. 
include(joinpath("train", "surrogate_models.jl"))
include(joinpath("train", "NODETrainParams.jl"))
include("constants.jl")
include(joinpath("generate_data", "fault_pvs.jl"))
include(joinpath("train", "HPCTrain.jl"))
include(joinpath("generate_data", "NODETrainInputs.jl"))
include(joinpath("train", "train.jl"))
include(joinpath("train", "multiple_shoot.jl"))
include("utils.jl")
end
