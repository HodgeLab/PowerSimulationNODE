function optimizer_map(key)
    d = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)
    return d[key]
end

function steadystate_solver_map(solver, tols)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => OrdinaryDiffEq.Tsit5,
    )
    return SteadyStateDiffEq.DynamicSS(d[solver](), abstol = tols[1], reltol = tols[2])
end

function solver_map(key)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => OrdinaryDiffEq.Tsit5,
    )
    return d[key]
end

function instantiate_steadystate_solver(inputs)
    return steadystate_solver_map(inputs.solver, inputs.tols)
end

function sensealg_map(key)
    d = Dict(
        "ForwardDiff" => GalacticOptim.AutoForwardDiff,
        "Zygote" => GalacticOptim.AutoZygote,
    )
    return d[key]
end

function activation_map(key)
    d = Dict("relu" => Flux.relu, "hardtanh" => Flux.hardtanh, "sigmoid" => Flux.sigmoid)
    return d[key]
end

function instantiate_solver(inputs)
    return solver_map(inputs.solver)()
end

function instantiate_sensealg(inputs)
    return sensealg_map(inputs.sensealg)()
end

function instantiate_optimizer(inputs)
    if inputs.optimizer.primary == "Adam"
        return optimizer_map(inputs.optimizer.primary)(inputs.optimizer.primary_η)
    elseif inputs.optimizer.primary == "Bfgs"
        return optimizer_map(inputs.optimizer.primary)()
    end
end

function instantiate_optimizer_adjust(inputs)
    if inputs.optimizer.adjust == "Adam"
        return optimizer_map(inputs.optimizer.adjust)(inputs.optimizer_adjust_η)
    elseif inputs.optimizer.adjust == "Bfgs"
        return optimizer_map(inputs.optimizer.adjust)()
    end
end

function instantiate_surrogate(TrainParams, n_ports)
    steadystate_solver = instantiate_steadystate_solver(TrainParams.steady_state_solver)
    dynamic_solver = instantiate_solver(TrainParams.dynamic_solver)
    model_initializer = _instantiate_model_initializer(TrainParams, n_ports)
    model_node = _instantiate_model_node(TrainParams, n_ports)
    model_observation = _instantiate_model_observation(TrainParams, n_ports)
    display(model_initializer)
    display(model_node)
    display(model_observation)
    dynamic_reltol = TrainParams.dynamic_solver.tols[1]
    dynamic_abstol = TrainParams.dynamic_solver.tols[2]
    dynamic_maxiters = TrainParams.dynamic_solver.maxiters
    steadystate_maxiters = TrainParams.steady_state_solver.maxiters

    return SteadyStateNeuralODE(
        model_initializer,
        model_node,
        model_observation,
        steadystate_solver,
        dynamic_solver,
        steadystate_maxiters;
        abstol = dynamic_abstol,
        reltol = dynamic_reltol,
        maxiters = dynamic_maxiters,
    )
end

function _instantiate_model_initializer(TrainParams, n_ports)
    initializer_params = TrainParams.model_initializer
    hidden_states = TrainParams.hidden_states
    type = initializer_params.type
    n_layer = initializer_params.n_layer
    width_layers = initializer_params.width_layers
    activation = activation_map(initializer_params.activation)

    vector_layers = []
    if type == "dense"
        push!(
            vector_layers,
            (x) -> (
                x .* TrainParams.input_normalization.x_scale .+
                TrainParams.input_normalization.x_bias
            ),
        )
        push!(
            vector_layers,
            Dense(SURROGATE_SS_INPUT_DIM * n_ports, width_layers, activation),
        )
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation))
        end
        push!(vector_layers, Dense(width_layers, hidden_states, activation))
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    elseif type == "OutputParams"
        @error "OutputParams layer for inititalizer not yet implemented"
    end
    return model
end

function _instantiate_model_node(TrainParams, n_ports)
    node_params = TrainParams.model_node
    hidden_states = TrainParams.hidden_states
    type = node_params.type
    n_layer = node_params.n_layer
    width_layers = node_params.width_layers
    activation = activation_map(node_params.activation)
    vector_layers = []
    if type == "dense"
        push!(
            vector_layers,
            Parallel(
                +,
                Chain(
                    (x) -> (
                        x .* TrainParams.input_normalization.exogenous_scale .+
                        TrainParams.input_normalization.exogenous_bias
                    ),
                    Dense(
                        SURROGATE_EXOGENOUS_INPUT_DIM * n_ports,
                        hidden_states,
                        activation,
                    ),
                ),
                Dense(hidden_states, hidden_states, activation),
            ),
        )
        if n_layer >= 1
            push!(vector_layers, Dense(hidden_states, width_layers, activation))
        end
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation))
        end
        push!(vector_layers, Dense(width_layers, hidden_states, activation))
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    end
    return model
end

function _instantiate_model_observation(TrainParams, n_ports)
    observation_params = TrainParams.model_observation
    hidden_states = TrainParams.hidden_states
    type = observation_params.type
    n_layer = observation_params.n_layer
    width_layers = observation_params.width_layers
    activation = activation_map(observation_params.activation)
    vector_layers = []
    if type == "dense"
        push!(vector_layers, Dense(hidden_states, width_layers, activation))
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation))
        end
        push!(
            vector_layers,
            Dense(width_layers, SURROGATE_OUTPUT_DIM * n_ports, activation),
        )
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    end
    return model
end

function _inner_loss_function(surrogate_solution, ground_truth_subset, params)
    rmse_weight = params.loss_function.type_weights.rmse
    mae_weight = params.loss_function.type_weights.mae
    A_weight = params.loss_function.component_weights.A
    B_weight = params.loss_function.component_weights.B
    C_weight = params.loss_function.component_weights.C
    r0_pred = surrogate_solution.r0_pred
    r0 = surrogate_solution.r0
    i_series = surrogate_solution.i_series
    ϵ = surrogate_solution.ϵ

    lossA =
        A_weight * (mae_weight * mae(r0_pred, r0) + rmse_weight * sqrt(mse(r0_pred, r0)))
    if size(ground_truth_subset) == size(i_series)
        lossB =
            B_weight * (
                mae_weight * mae(ground_truth_subset, i_series) +
                rmse_weight * sqrt(mse(ground_truth_subset, i_series))
            )
    else
        lossB = Inf
    end

    lossC =
        C_weight * (
            mae_weight * mae(ϵ, zeros(length(ϵ))) +
            rmse_weight * sqrt(mse(ϵ, zeros(length(ϵ))))
        )
    return lossA, lossB, lossC
end

function instantiate_outer_loss_function(
    surrogate::SteadyStateNeuralODE,
    fault_data::Array{TrainData},
    branch_order::Array{String},
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
)
    return (θ, fault_timespan_index) -> _outer_loss_function(
        θ,
        fault_timespan_index,
        surrogate,
        fault_data,
        branch_order,
        exs,
        train_details,
        params,
    )
end

function _outer_loss_function(
    θ,
    fault_timespan_index::Tuple{Int64, Int64},
    surrogate::SteadyStateNeuralODE,
    fault_data::Array{TrainData},
    branch_order::Array{String},
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
)
    fault_index = fault_timespan_index[1]
    timespan_index = fault_timespan_index[2]
    ex = exs[fault_index]
    powerflow = fault_data[fault_index].powerflow
    tsteps = fault_data[fault_index].tsteps
    groundtruth_current = fault_data[fault_index].groundtruth_current
    index_subset = _find_subset(tsteps, train_details[timespan_index])
    t_subset = tsteps[index_subset]
    groundtruth_subset = groundtruth_current[:, index_subset]
    surrogate_solution = surrogate(ex, powerflow, t_subset, θ)
    lossA, lossB, lossC =
        _inner_loss_function(surrogate_solution, groundtruth_subset, params)
    return lossA + lossB + lossC,
    lossA,
    lossB,
    lossC,
    surrogate_solution,
    fault_timespan_index
end

function _find_subset(tsteps, train_details)
    tspan = train_details.tspan
    batching_sample_factor = train_details.batching_sample_factor
    subset = BitArray([
        t >= tspan[1] && t <= tspan[2] && rand(1)[1] < batching_sample_factor for
        t in tsteps
    ])
    return subset
end

function instantiate_cb!(output, lb_loss, exportmode_skip)
    if Sys.iswindows() || Sys.isapple()
        print_loss = true
    else
        print_loss = false
    end

    return (p, l, lA, lB, lC, surrogate_solution, fault_index) -> _cb!(
        p,
        l,
        lA,
        lB,
        lC,
        surrogate_solution,
        fault_index,
        output,
        lb_loss,
        print_loss,
        exportmode_skip,
    )
end

function _cb!(
    p,
    l,
    lA,
    lB,
    lC,
    surrogate_solution,
    fault_index,
    output,
    lb_loss,
    print_loss,
    exportmode_skip,
)
    push!(output["loss"], (lA, lB, lC, l))
    output["total_iterations"] += 1
    if mod(output["total_iterations"], exportmode_skip) == 0
        push!(output["predictions"], ([p], surrogate_solution, fault_index))
        push!(output["recorded_iterations"], output["total_iterations"])
    end

    if (print_loss)
        println(l)
    end
    (l > lb_loss) && return false
    return true
end
