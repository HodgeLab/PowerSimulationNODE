"""
    mutable struct TrainParams

# Fields
- `train_id::String`: id for the training instance, used for naming output data folder.
- `surrogate_buses::Vector{Int64}`: The buses which make up the portion of the system to be replaced with a surrogate.
- `train_data::NamedTuple{
    (:id, :operating_points, :perturbations, :params, :system),
    Tuple{
        String,
        Vector{PSIDS.SurrogateOperatingPoint},
        Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
        PSIDS.GenerateDataParams,
        String,
    },
    }`: train_data describes the training dataset. The `:system` field options are `"reduced"` and `"full"`. 
- `validation_data::NamedTuple{
    (:id, :operating_points, :perturbations, :params),
    Tuple{
        String,
        Vector{PSIDS.SurrogateOperatingPoint},
        Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
        PSIDS.GenerateDataParams,
    },
    }`: validation_data describes the validation dataset. No system option because validation data always comes from full system. 

- `test_data::NamedTuple{
    (:id, :operating_points, :perturbations, :params),
    Tuple{
        String,
        Vector{PSIDS.SurrogateOperatingPoint},
        Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
        PSIDS.GenerateDataParams,
    },
    }`: test_data describes the validation dataset. No system option because test data always comes from full system. 
- `hidden_states::Int64`: The number of surrogate states. User defined depending on complexity of underlying model.
- `model_initializer::NamedTuple{
    (:type, :n_layer, :width_layers, :activation),
    Tuple{String, Int64, Int64, String},
    }`: Parameters which determine the structure of the initializer NN. `type="dense"`. `n_layer` is the number of hidden layers. `width_layers` is the width of hidden layers. `activation=["tanh", "relu"]` in the activation function 
- `model_node::NamedTuple{
    (
        :type,
        :n_layer,
        :width_layers,
        :activation,
        :σ2_initialization,
    ),
    Tuple{String, Int64, Int64, String, Float64},
    }`: Parameters which determine the structure of the neural ODE. `type="dense"`. `n_layer` is the number of hidden layers. `width_layers` is the width of hidden layers. `activation=["tanh", "relu"]` in the activation function. 
    `σ2_initialization` is the variance of the initial params for the node model. Set `σ2_initialization = 0.0` to use the default flux initialization.
- `model_observation::NamedTuple{
    (:type, :n_layer, :width_layers, :activation, :normalization),
    Tuple{String, Int64, Int64, String, String},
    }`: Parameters which determine the structure of the observer NN. `type="dense"`. `n_layer` is the number of hidden layers. `width_layers` is the width of hidden layers. `activation=["tanh", "relu"]` in the activation function. 
- `input_normalization::NamedTuple{
    (:x_scale, :x_bias, :exogenous_scale, :exogenous_bias),
    Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}},
    }`: Scale and bias parameters for the exogenous input and the fixed input. 
- `steady_state_solver::NamedTuple{
    (:solver, :tols),
    Tuple{String, Tuple{Float64, Float64}},
    }`: The solver used for initializing surrogate to steady state. `solver` is the solver name from DifferentialEquations.jl, `tols` is a tuple `(abstol, reltol)`
- `dynamic_solver::NamedTuple{
    (:solver, :tols),
    Tuple{String, Tuple{Float64, Float64}},
    }`: The solver used for solving the neural ODE dynamics. `solver` is the solver name from DifferentialEquations.jl, `tols` is a tuple `(abstol, reltol)`
- `optimizer::NamedTuple{
    (:sensealg, :primary, :primary_η, :adjust, :adjust_η),
    Tuple{String, String, Float64, String, Float64},
    }`: The optimizer(s) used during training. `sensealg="AutoZygote"`. The primary optimizer is used throughout the training according to the data provided and the `curriculum`/`curriculum_timespans` parameter.
    WARNING: The adjust optimizer is not yet implemented (TODO)
- `p_start::Vector{Float32}`: Starting parameters (for initializer, node, and observation together). By default is empty which starts with randomly initialized parameters (see `rng_seed`). 
- `maxiters::Int64`: The maximum possible iterations for the entire training instance. If `lb_loss = 0` and `optimizer = "Adam"` the training should never exit early and maxiters will be hit.
    Note that the number of saved data points can exceed maxiters because there is an additional callback at the end of each individual optimization.
- `lb_loss::Float64`: If the value of the loss function moves below lb_loss during training, the current optimization ends (current range).
- `curriculum::String`: A curriculum for ordering the training data. `none` will train on all of the data simultaneously.  `progressive` will train on a single fault before moving to the next fault. 
- `curriculum_timespans::Array{
    NamedTuple{
        (:tspan, :batching_sample_factor),
        Tuple{Tuple{Float64, Float64}, Float64},
    },
    }`: Indicates the timespan to train on and the sampling factor for batching. If more than one entry in `curriculum_timespans`, each fault from the input data is paired with each value of `curriculum_timespans`
- `validation_loss_every_n::Int64`: Determines how often, during training, the surrogate is added to the full system and loss is evaluated. 
- `loss_function::NamedTuple{
    (:component_weights, :type_weights),
    Tuple{
        NamedTuple{(:initialization_weight, :dynamic_weight, :residual_penalty), Tuple{Float64, Float64, Float64}},
        NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
    }, 
    }`: Various weighting factors to change the loss function. `initialization_weight` scales the portion of loss function penalizing the difference between predicted steady state and actual.
    `dynamic_weight` scales the portion of the loss function penalizing differences in the output time series.
    `residual_penalty` scales the loss function if the implicit layer does not converge (if set to Inf, the loss is infinite anytime the implicit layer does not converge).
    `type_weights` should sum to one. 
- `rng_seed::Int64`: Seed for the random number generator used for initializing the NN for reproducibility across training runs.
- `output_mode_skip::Int`: Record and save output data every `output_mode_skip` iterations. Meant to ease memory constraints on HPC. 
- `train_time_limit_seconds::Int64`:  
- `base_path:String`: TODO: Directory for training where input data is found and output data is written.
- `system_path::String`: Location of the full `System`. Training/validation/test systems are dervied based on `surrogate_buses`.  
- `surrogate_system_path`: Path to validation system (surrogate is added to this system during training).
- `train_system_path`: Path to train system (system with only the surrogate represented in detail with sources surrounding).
- `train_data_path::String`: path to train data. 
- `validation_data_path::String`: path to validation data.
- `test_data_path::String`: path to test_data. 
- `output_data_path:String`: From `base_path`, the directory for saving output data.
- `force_gc:Bool`: `true`: After training and before writing outputs to file, force GC.gc() in order to alleviate out-of-memory problems on hpc.
"""
mutable struct TrainParams
    train_id::String
    surrogate_buses::Vector{Int64}
    train_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params, :system),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
            String,
        },
    }
    validation_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
        },
    }
    test_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
        },
    }
    hidden_states::Int64
    model_initializer::NamedTuple{
        (:type, :n_layer, :width_layers, :activation),
        Tuple{String, Int64, Int64, String},
    }
    model_node::NamedTuple{
        (:type, :n_layer, :width_layers, :activation, :σ2_initialization),
        Tuple{String, Int64, Int64, String, Float64},
    }
    model_observation::NamedTuple{
        (:type, :n_layer, :width_layers, :activation),
        Tuple{String, Int64, Int64, String},
    }
    input_normalization::NamedTuple{
        (:x_scale, :x_bias, :exogenous_scale, :exogenous_bias),
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}},
    }
    steady_state_solver::NamedTuple{
        (:solver, :abstol, :maxiters),
        Tuple{String, Float64, Int},
    }
    dynamic_solver::NamedTuple{
        (:solver, :tols, :maxiters),
        Tuple{String, Tuple{Float64, Float64}, Int},
    }
    optimizer::NamedTuple{
        (:sensealg, :primary, :primary_η, :adjust, :adjust_η),
        Tuple{String, String, Float64, String, Float64},
    }
    p_start::Vector{Float32}
    maxiters::Int64
    lb_loss::Float64
    curriculum::String
    curriculum_timespans::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    }
    validation_loss_every_n::Int64
    loss_function::NamedTuple{
        (:component_weights, :type_weights),
        Tuple{
            NamedTuple{
                (:initialization_weight, :dynamic_weight, :residual_penalty),
                Tuple{Float64, Float64, Float64},
            },
            NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
        },
    }
    rng_seed::Int64
    output_mode_skip::Int64
    train_time_limit_seconds::Int64
    base_path::String
    system_path::String
    surrogate_system_path::String
    train_system_path::String
    connecting_branch_names_path::String
    train_data_path::String
    validation_data_path::String
    test_data_path::String
    output_data_path::String
    force_gc::Bool
end

StructTypes.StructType(::Type{TrainParams}) = StructTypes.Mutable()
StructTypes.StructType(::Type{PSIDS.GenerateDataParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.PVS}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.VStep}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.GenerationLoadScale}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomBranchTrip}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomLoadTrip}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomLoadChange}) = StructTypes.Struct()

StructTypes.StructType(::Type{PSIDS.SurrogatePerturbation}) = StructTypes.AbstractType()
StructTypes.StructType(::Type{PSIDS.SurrogateOperatingPoint}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{PSIDS.SurrogatePerturbation}) = :type
StructTypes.subtypekey(::Type{PSIDS.SurrogateOperatingPoint}) = :type
StructTypes.subtypes(::Type{PSIDS.SurrogatePerturbation}) = (
    PVS = PSIDS.PVS,
    VStep = PSIDS.VStep,
    RandomBranchTrip = PSIDS.RandomBranchTrip,
    RandomLoadTrip = PSIDS.RandomLoadTrip,
    RandomLoadChange = PSIDS.RandomLoadChange,
)
StructTypes.subtypes(::Type{PSIDS.SurrogateOperatingPoint}) =
    (GenerationLoadScale = PSIDS.GenerationLoadScale,)

function TrainParams(;
    train_id = "train_instance_1",
    surrogate_buses = [1],
    train_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.PVS(source_name = "InfBus")]],
        params = PSIDS.GenerateDataParams(),
        system = "reduced",     #generate from the reduced system with sources to perturb or the full system
    ),
    validation_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.VStep(source_name = "InfBus")]],    #To do - make this a branch impedance double 
        params = PSIDS.GenerateDataParams(),
    ),
    test_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.VStep(source_name = "InfBus")]],    #To do - make this a branch impedance double 
        params = PSIDS.GenerateDataParams(),
    ),
    hidden_states = 5,
    model_initializer = (
        type = "dense",     #OutputParams (train initial conditions)
        n_layer = 0,
        width_layers = 4,
        activation = "hardtanh",
    ),
    model_node = (
        type = "dense",
        n_layer = 1,
        width_layers = 4,
        activation = "hardtanh",
        σ2_initialization = 0.0,
    ),
    model_observation = (
        type = "dense",
        n_layer = 0,
        width_layers = 4,
        activation = "hardtanh",
    ),
    input_normalization = (
        x_scale = [1.0, 1.0, 1.0],
        x_bias = [0.0, 0.0, 0.0],
        exogenous_scale = [1.0, 1.0],
        exogenous_bias = [0.0, 0.0],
    ),
    steady_state_solver = (
        solver = "SSRootfind",
        abstol = 1e-4,       #xtol, ftol  #High tolerance -> standard NODE with initializer and observation 
        maxiters = 5,
    ),
    dynamic_solver = (solver = "Rodas4", tols = (1e-6, 1e-6), maxiters = 1e3),
    optimizer = (
        sensealg = "Zygote",
        primary = "Adam",
        primary_η = 0.000001,
        adjust = "nothing",
        adjust_η = 0.0,
    ),
    p_start = [],
    maxiters = 15,
    lb_loss = 0.0,
    curriculum = "none",
    curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
    validation_loss_every_n = 100,
    loss_function = (
        component_weights = (
            initialization_weight = 1.0,
            dynamic_weight = 1.0,
            residual_penalty = 1.0,
        ),
        type_weights = (rmse = 1.0, mae = 0.0),
    ),
    rng_seed = 123,
    output_mode_skip = 1,
    train_time_limit_seconds = 1e9,
    base_path = pwd(),
    system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "system.json",
    ),
    surrogate_system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "validation_system.json",
    ),
    train_system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "train_system.json",
    ),
    connecting_branch_names_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "connecting_branches_names",
    ),
    train_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "train_data_$(train_data.id)",
    ),
    validation_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "validation_data_$(validation_data.id)",
    ),
    test_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "test_data_$(test_data.id)",
    ),
    output_data_path = joinpath(base_path, "output_data"),
    force_gc = true,
)
    TrainParams(
        train_id,
        surrogate_buses,
        train_data,
        validation_data,
        test_data,
        hidden_states,
        model_initializer,
        model_node,
        model_observation,
        input_normalization,
        steady_state_solver,
        dynamic_solver,
        optimizer,
        p_start,
        maxiters,
        lb_loss,
        curriculum,
        curriculum_timespans,
        validation_loss_every_n,
        loss_function,
        rng_seed,
        output_mode_skip,
        train_time_limit_seconds,
        base_path,  #other paths derived from this one must come after 
        system_path,
        surrogate_system_path,
        train_system_path,
        connecting_branch_names_path,
        train_data_path,
        validation_data_path,
        test_data_path,
        output_data_path,
        force_gc,
    )
end

function TrainParams(file::AbstractString)
    return JSON3.read(read(file), TrainParams)
end

"""
    serialize(inputs::TrainParams, file_path::String)

Serializes  the input to JSON file.
"""
function serialize(inputs::TrainParams, file_path::String)
    open(file_path, "w") do io
        JSON3.write(io, inputs)
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", params::TrainParams)
    for field_name in fieldnames(TrainParams)
        field = getfield(params, field_name)
        if isa(field, NamedTuple)
            println(io, "$field_name =")
            for k in keys(field)
                println(io, "\t", k, " : ", field[k])
            end
        else
            println(io, "$field_name = ", getfield(params, field_name))
        end
    end
end
