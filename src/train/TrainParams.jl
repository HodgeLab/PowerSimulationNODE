"""
    mutable struct TrainParams

# Fields
- `train_id::String`: id for the training instance, used for naming output data folder.
- `hidden_states::Int64`: TODO - DESCRIPTION
- `model_initializer::NamedTuple{
    (:type, :n_layer, :width_layers, :activation, :normalization),
    Tuple{String, Int64, Int64, String, String},
    }`: TODO - DESCRIPTION
- `model_node::NamedTuple{
    (
        :type,
        :n_layer,
        :width_layers,
        :activation,
        :normalization,
        :initialization,
        :exogenous_input,
    ),
    Tuple{String, Int64, Int64, String, String, String, String},
    }`: TODO - DESCRIPTION
- `model_observation::NamedTuple{
    (:type, :n_layer, :width_layers, :activation, :normalization),
    Tuple{String, Int64, Int64, String, String},
    }`: TODO - DESCRIPTION
- `steady_state_solver::NamedTuple{
    (:solver, :tols, :sensealg),
    Tuple{String, Tuple{Float64, Float64}, String},
    }`: TODO - DESCRIPTION
- `dynamic_solver::NamedTuple{
    (:solver, :tols, :sensealg),
    Tuple{String, Tuple{Float64, Float64}, String},
    }`: TODO - DESCRIPTION
- `optimizer::NamedTuple{
    (:primary, :primary_η, :adjust, :adjust_η),
    Tuple{String, Float64, String, Float64},
    }`: TODO - DESCRIPTION 
- `maxiters::Int64`: The maximum possible iterations for the entire training instance. If `lb_loss = 0` and `optimizer = "Adam"` the training should never exit early and maxiters will be hit.
    Note that the number of saved data points can exceed maxiters because there is an additional callback at the end of each individual optimization.
- `lb_loss::Float64`: If the value of the loss function moves below lb_loss during training, the current optimization ends (current range).
- `curriculum::String`: TODO - DESCRIPTION 
- `curriculum_timespans::Array{
    NamedTuple{
        (:tspan, :batching_sample_factor),
        Tuple{Tuple{Float64, Float64}, Float64},
    },
    }`: TODO - DESCRIPTION 
- `loss_function::NamedTuple{
    (:weights, :scale),
    Tuple{Tuple{Float64, Float64, Float64}, String},
    }`: TODO - DESCRIPTION
- `rng_seed::Int64`: Seed for the random number generator used for initializing the NN for reproducibility across training runs.
- `output_mode::Int`: `1`: do not collect any data during training, only save high-level data related to training and final results `2`: Same as `1`, also save value of loss throughout training. Valid values [1,2,3]
    `3`: same as `2`, also save parameters and predictions during training.
- `output_mode_skip::Int`: Only matters if output_mode = 3. Record paramters and predictions every n iterations. Meant to ease memory constraints on HPC. 
- `base_path:String`: Directory for training where input data is found and output data is written.
- `input_data_path:String`: From `base_path`, the directory for input data.
- `output_data_path:String`: From `base_path`, the directory for saving output data.
- `force_gc:Bool`: `true`: After training and before writing outputs to file, force GC.gc() in order to alleviate out-of-memory problems on hpc.
"""
mutable struct TrainParams
    train_id::String
    hidden_states::Int64
    model_initializer::NamedTuple{
        (:type, :n_layer, :width_layers, :activation, :normalization),
        Tuple{String, Int64, Int64, String, String},
    }
    model_node::NamedTuple{
        (
            :type,
            :n_layer,
            :width_layers,
            :activation,
            :normalization,
            :initialization,
            :exogenous_input,
        ),
        Tuple{String, Int64, Int64, String, String, String, String},
    }
    model_observation::NamedTuple{
        (:type, :n_layer, :width_layers, :activation, :normalization),
        Tuple{String, Int64, Int64, String, String},
    }
    steady_state_solver::NamedTuple{
        (:solver, :tols, :sensealg),
        Tuple{String, Tuple{Float64, Float64}, String},
    }
    dynamic_solver::NamedTuple{
        (:solver, :tols, :sensealg, :maxiters),
        Tuple{String, Tuple{Float64, Float64}, String, Int},
    }
    optimizer::NamedTuple{
        (:sensealg, :primary, :primary_η, :adjust, :adjust_η),
        Tuple{String, String, Float64, String, Float64},
    }
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
        (:component_weights, :type_weights, :scale),
        Tuple{
            NamedTuple{(:A, :B, :C), Tuple{Float64, Float64, Float64}},
            NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
            String,
        },
    }
    rng_seed::Int64
    output_mode::Int64
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
        normalization = "default",
    ),
    model_node = (
        type = "dense",
        n_layer = 1,
        width_layers = 4,
        activation = "hardtanh",
        normalization = "default",
        initialization = "default",
        exogenous_input = "V",
    ),
    model_observation = (
        type = "dense",
        n_layer = 0,
        width_layers = 4,
        activation = "hardtanh",
        normalization = "default",
    ),
    steady_state_solver = (
        solver = "Tsit5",
        tols = (1e-4, 1e-4),        #High tolerance -> standard NODE with initializer and observation 
        sensealg = "InterpolatingAdjoint",
    ),
    dynamic_solver = (
        solver = "Tsit5",
        tols = (1e-6, 1e-6),
        sensealg = "InterpolatingAdjoint",
        maxiters = 1e3,
    ),
    optimizer = (
        sensealg = "Zygote",
        primary = "Adam",
        primary_η = 0.0001,
        adjust = "nothing",
        adjust_η = 0.0,
    ),
    maxiters = 15,
    lb_loss = 0.0,
    curriculum = "none", #"sequential" 
    curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
    loss_function = (
        component_weights = (A = 1.0, B = 1.0, C = 1.0),
        type_weights = (rmse = 1.0, mae = 0.0),
        scale = "default",
    ),
    rng_seed = 123,
    output_mode = 3,
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
        steady_state_solver,
        dynamic_solver,
        optimizer,
        maxiters,
        lb_loss,
        curriculum,
        curriculum_timespans,
        loss_function,
        rng_seed,
        output_mode,
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
