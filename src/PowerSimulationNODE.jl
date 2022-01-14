module PowerSimulationNODE

#export the structs / functions I currently use in the scripts
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

#TODO - Change to import for clarity of namespace
import Arrow
import Optim
import DataFrames    
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
import OrdinaryDiffEq    
using PowerSimulationsDynamics
using PowerSystems
import Random
import StructTypes
import YAML

const PSY = PowerSystems
const PSID = PowerSimulationsDynamics

using Requires
function __init__()
    #@require DiffEqSensitivity = "41bf760c-e81c-5289-8e54-58b1f1f8abe2" include("instantiate.jl")  #instantiations called from other files 
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("visualize.jl")
end
include("surrogate_models.jl")
include("NODETrainParams.jl")
include("constants.jl")
include("fault_pvs.jl")
include("HPCTrain.jl")
include("instantiate.jl")
include("NODETrainInputs.jl")
include("train.jl")
include("utils.jl")
end
