"""
    mutable struct NODETrainParams

# Fields
- `train_id::Int64`: id for the training instance, used for naming output data folder.
- `solver::String`: solver used for the NODE problem. Valid Values ["Rodas4", "Tsit5", "TRBDF2"]
- `solver_tols:: Tuple{Float64, Float64}`: solver tolerances (abstol, reltol).
- `solver_sensealg::String`: sensitivity algorithm for the ODE solve ["InterpolatingAdjoint", "InterpolatingAdjoint_checkpointing"]
- `sensealg::String`: sensitivity algorithm used in training. Valid Values ["ForwardDiff", "Zygote" ]
- `optimizer::["Adam", "Bfgs"]`: main optimizer used in training.
- `optimizer_η::Float64`: Learning rate for Adam (amount by which gradients are discounted before updating weights). Ignored if Adam is not the optimizer.
- `optimizer_adjust::String: optimizer used for final adjustments (2nd stage). Valid values ["Adam", "Bfgs", "nothing"]. NOT YET IMPLEMENTED (TODO)
- `optimizer_adjust_η::Float64`: Learning rate for Adam (amount by which gradients are discounted before updating weights). Ignored if Adam is not the optimizer.  NOT YET IMPLEMENTED (TODO)
- `maxiters::Int64`: The maximum possible iterations for the entire training instance. If `lb_loss = 0` and `optimizer = "Adam"` the training should never exit early and maxiters will be hit.
    Note that the number of saved data points can exceed maxiters because there is an additional callback at the end of each individual optimization.
- `lb_loss::Float64`: If the value of the loss function moves below lb_loss during training, the current optimization ends (current range).
- `training_groups::DataStructures.SortedDict{
    Tuple{Float64, Float64},
    NamedTuple{
        (:shoot_times, :multiple_shoot_continuity_term, :batching_sample_factor),
        Tuple{Int64, Tuple{Float64, Float64}, Float64},
    }`: Specify the tspan for each group of training, and the multiple shooting and random batching parameter for each group.  
- `groupsize_faults::Int64`: Number of faults trained on simultaneous `1`:sequential training. if equal to number of pvs in sys_train, parallel training.
- `loss_function_weights::Tuple{Float64, Float64}`: weights used for loss function `(mae_weight, mse_weight)`.
- `loss_function_scale::String`: Scaling of the loss function.  `"range"`: the range of the real current and imaginary current are used to scale both the mae. Valid values ["range", "none"]
    and mse portions of the loss function. The goal is to give equal weight to real and imaginary components even if the magnitude of the disturbance differs. `"none"`: no additional scaling applied.
- `ode_model::String ["none","vsm"]`: The ode model used in conjunction with the NODE during training. `"none"` uses a purely data driven NODE surrogate. 
- `node_input_scale::Float64`: Scale factor on the voltage input to the NODE. Does not apply to other inputs (ie the feedback states).
- `node_output_scale::Float64`: Scale factor on the current output of the NODE. Does not apply to other outputs (ie the feedback states).
- `node_state_inputs::Vector{Tuple{String, Symbol`: Additional states that are input to NODE: ("DeviceName", :StateSymbol). The device name and symbol must match the solution objected passed as input data. 
- `observation_function::String`: Function from latent space of NODE to observed states.  Valid Values ["first_n"]
- `node_unobserved_states::Int64`: Number of feedback states in the NODE. Does not include the output current states which can be feedback if `node_feedback_current = true`.
- `node_feedback_current::Bool`: Determines if current is also a feedback state.
- `node_layers::Int64`: Number of hidden layers in the NODE. Does not include the input or output layer.
- `node_width::Int64`: Number of neurons in each hidden layer. Each hidden layer has the same number of neurons. The width of the input and output layers are determined by the combination of other parameters.
- `node_activation::String`: Activation function for NODE. The output layer always uses the identity activation. Valid Values ["relu", "hardtanh", "sigmoid"]
- `rng_seed::Int64`: Seed for the random number generator used for initializing the NN for reproducibility across training runs.
- `output_mode::Int`: `1`: do not collect any data during training, only save high-level data related to training and final results `2`: Same as `1`, also save value of loss throughout training. Valid values [1,2,3]
    `3`: same as `2`, also save parameters and predictions during training.
- `output_mode_skip::Int`: Only matters if output_mode = 3. Record paramters and predictions every n iterations. Meant to ease memory constraints on HPC. 
- `base_path:String`: Directory for training where input data is found and output data is written.
- `input_data_path:String`: From `base_path`, the directory for input data.
- `output_data_path:String`: From `base_path`, the directory for saving output data.
- `verify_psid_node_off:Bool`: `true`: before training, check that the surrogate with NODE turned off matches the data provided from PSID simulation.
"""
mutable struct NODETrainParams
    train_id::String
    solver::String
    solver_tols::Tuple{Float64, Float64}
    solver_sensealg::String
    sensealg::String
    optimizer::String
    optimizer_η::Float64
    optimizer_adjust::String
    optimizer_adjust_η::Float64
    maxiters::Int64
    lb_loss::Float64
    training_groups::Array{
        NamedTuple{
            (
                :tspan,
                :shoot_times,
                :multiple_shoot_continuity_term,
                :batching_sample_factor,
            ),
            Tuple{
                Tuple{Float64, Float64},
                Array{Float64},
                Tuple{Float64, Float64},
                Float64,
            },
        },
    }
    groupsize_faults::Int64
    loss_function_weights::Tuple{Float64, Float64}
    loss_function_scale::String
    ode_model::String
    input_PQ::Bool
    node_input_scale::Float64
    node_output_scale::Float64
    node_state_inputs::Vector{Tuple{String, Symbol}}
    observation_function::String
    node_unobserved_states::Int64   #changed from node_unobserved_states
    initialize_unobserved_states::String
    learn_initial_condition_unobserved_states::Bool
    node_layers::Int64
    node_width::Int64
    node_activation::String
    rng_seed::Int64
    output_mode::Int64
    output_mode_skip::Int64
    base_path::String
    input_data_path::String
    output_data_path::String
    verify_psid_node_off::Bool
end

StructTypes.StructType(::Type{NODETrainParams}) = StructTypes.Mutable()

function NODETrainParams(;
    train_id = "train_instance_1",
    solver = "Rodas4",
    solver_tols = (1e-6, 1e-3),
    solver_sensealg = "InterpolatingAdjoint",
    sensealg = "ForwardDiff",
    optimizer = "Adam",
    optimizer_η = 0.001,
    optimizer_adjust = "nothing",
    optimizer_adjust_η = 0.001,
    maxiters = 15,
    lb_loss = 0.0,
    training_groups = [(
        tspan = (0.0, 1.0),
        shoot_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        multiple_shoot_continuity_term = (100.0, 100.0),
        batching_sample_factor = 1.0,
    )],
    groupsize_faults = 1,
    loss_function_weights = (1.0, 0.0),
    loss_function_scale = "none",
    ode_model = "none",
    input_PQ = false,
    node_input_scale = 10e1,
    node_output_scale = 1.0,
    node_state_inputs = [],
    observation_function = "first_n",
    node_unobserved_states = 0,
    initialize_unobserved_states = "random",
    learn_initial_condition_unobserved_states = false,
    node_layers = 2,
    node_width = 2,
    node_activation = "relu",
    rng_seed = 1234,
    output_mode = 3,
    output_mode_skip = 1,
    base_path = pwd(),
    input_data_path = joinpath(base_path, "input_data"),
    output_data_path = joinpath(base_path, "output_data"),
    verify_psid_node_off = true,
)
    NODETrainParams(
        train_id,
        solver,
        solver_tols,
        solver_sensealg,
        sensealg,
        optimizer,
        optimizer_η,
        optimizer_adjust,
        optimizer_adjust_η,
        maxiters,
        lb_loss,
        training_groups,
        groupsize_faults,
        loss_function_weights,
        loss_function_scale,
        ode_model,
        input_PQ,
        node_input_scale,
        node_output_scale,
        node_state_inputs,
        observation_function,
        node_unobserved_states,
        initialize_unobserved_states,
        learn_initial_condition_unobserved_states,
        node_layers,
        node_width,
        node_activation,
        rng_seed,
        output_mode,
        output_mode_skip,
        base_path,
        input_data_path,
        output_data_path,
        verify_psid_node_off,
    )
end

function NODETrainParams(file::AbstractString)
    return JSON3.read(read(file), NODETrainParams)
end

"""
    serialize(inputs::NODETrainParams, file_path::String)

Serializes  the input to JSON file.
"""
function serialize(inputs::NODETrainParams, file_path::String)
    open(file_path, "w") do io
        JSON3.write(io, inputs)
    end
    return
end
