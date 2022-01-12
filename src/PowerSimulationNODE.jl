module PowerSimulationNODE

#export the structs / functions I currently use in the scripts
export train
export NODETrainParams
export serialize
export node_load_system
export SummitHPCTrain
export SavioHPCTrain
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

#Change to import for clarit of namespace
using Requires
import Mustache          #Render 
using OrdinaryDiffEq     #Rodas4
using DiffEqSensitivity
using Logging
using PowerSystems
using PowerSimulationsDynamics
using GalacticOptim
using DiffEqFlux    #BFGS     
using Flux
using Flux.Losses: mae, mse

import Arrow
import YAML
import StructTypes
const PSY = PowerSystems
const PSID = PowerSimulationsDynamics

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("visualize.jl")
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
