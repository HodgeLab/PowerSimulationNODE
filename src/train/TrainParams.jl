"""
    mutable struct TrainParams

# Fields
- `train_id::String`: id for the training instance, used for naming output data folder.
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
        :initialization,
    ),
    Tuple{String, Int64, Int64, String, String},
    }`: Parameters which determine the structure of the neural ODE. `type="dense"`. `n_layer` is the number of hidden layers. `width_layers` is the width of hidden layers. `activation=["tanh", "relu"]` in the activation function. 
    WARNING: Custom initialization routines not yet implemented (TODO)
- `model_observation::NamedTuple{
    (:type, :n_layer, :width_layers, :activation, :normalization),
    Tuple{String, Int64, Int64, String, String},
    }`: Parameters which determine the structure of the observer NN. `type="dense"`. `n_layer` is the number of hidden layers. `width_layers` is the width of hidden layers. `activation=["tanh", "relu"]` in the activation function. 
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
- `    loss_function::NamedTuple{
    (:component_weights, :type_weights),
    Tuple{
        NamedTuple{(:A, :B, :C), Tuple{Float64, Float64, Float64}},
        NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
    },
    }`: Various weighting factors to change the loss function. For A,B,C definitions -- see paper. `type_weights` should sum to one. 
- `rng_seed::Int64`: Seed for the random number generator used for initializing the NN for reproducibility across training runs.
- `output_mode_skip::Int`: Record and save output data every `output_mode_skip` iterations. Meant to ease memory constraints on HPC. 
- `base_path:String`: Directory for training where input data is found and output data is written.
- `input_data_path:String`: From `base_path`, the directory for input data.
- `output_data_path:String`: From `base_path`, the directory for saving output data.
- `force_gc:Bool`: `true`: After training and before writing outputs to file, force GC.gc() in order to alleviate out-of-memory problems on hpc.
"""
mutable struct TrainParams
    train_id::String
    hidden_states::Int64
    model_initializer::NamedTuple{
        (:type, :n_layer, :width_layers, :activation),
        Tuple{String, Int64, Int64, String},
    }
    model_node::NamedTuple{
        (
            :type,
            :n_layer,
            :width_layers,
            :activation,
            :initialization,    #TODO - implement 
        ),
        Tuple{String, Int64, Int64, String, String},
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
        (:solver, :tols, :maxiters),
        Tuple{String, Tuple{Float64, Float64}, Int},
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
    loss_function::NamedTuple{
        (:component_weights, :type_weights),
        Tuple{
            NamedTuple{(:A, :B, :C), Tuple{Float64, Float64, Float64}},
            NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
        },
    }
    rng_seed::Int64
    output_mode_skip::Int64
    base_path::String
    input_data_path::String
    output_data_path::String
    force_gc::Bool
end

StructTypes.StructType(::Type{TrainParams}) = StructTypes.Mutable()

function TrainParams(;
    train_id = "train_instance_1",
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
        initialization = "default",
    ),
    model_observation = (
        type = "dense",
        n_layer = 0,
        width_layers = 4,
        activation = "hardtanh",
    ),
    input_normalization = (
        x_scale = [1.0, 1.0, 1.0, 1.0],
        x_bias = [0.0, 0.0, 0.0, 0.0],
        exogenous_scale = [1.0, 1.0],
        exogenous_bias = [0.0, 0.0],
    ),
    steady_state_solver = (
        solver = "Tsit5",
        tols = (1e-4, 1e-4),        #High tolerance -> standard NODE with initializer and observation 
        maxiters = 1e3,
    ),
    dynamic_solver = (solver = "Tsit5", tols = (1e-6, 1e-6), maxiters = 1e3),
    optimizer = (
        sensealg = "Zygote",
        primary = "Adam",
        primary_η = 0.0001,
        adjust = "nothing",
        adjust_η = 0.0,
    ),
    p_start = [],
    maxiters = 15,
    lb_loss = 0.0,
    curriculum = "none",
    curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
    loss_function = (
        component_weights = (A = 1.0, B = 1.0, C = 1.0),
        type_weights = (rmse = 1.0, mae = 0.0),
    ),
    rng_seed = 123,
    output_mode_skip = 1,
    base_path = pwd(),
    input_data_path = joinpath(base_path, "input_data"),
    output_data_path = joinpath(base_path, "output_data"),
    force_gc = true,
)
    TrainParams(
        train_id,
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
        loss_function,
        rng_seed,
        output_mode_skip,
        base_path,
        input_data_path,
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

import Base.==
function ==(x::TrainParams, y::TrainParams)
    for f in fieldnames(TrainParams)
        if getfield(x, f) != getfield(y, f)
            return false
        end
    end
    return true
end
