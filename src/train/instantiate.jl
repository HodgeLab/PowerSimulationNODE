function optimizer_map(key)
    d = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)
    return d[key]
end

function NormalInitializer(μ = 0.0f0, σ² = 0.01f0)
    return (dims...) -> randn(Float32, dims...) .* σ² .+ μ
end

function steadystate_solver_map(solver, tols)
    d = Dict(
        "Rodas4" => SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.Rodas4()),
        #"TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.Tsit5()),
        "SSRootfind" => SteadyStateDiffEq.SSRootfind(),
    )
    return d[solver]
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
    return steadystate_solver_map(inputs.solver, inputs.abstol)
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

function instantiate_surrogate_psid(
    params::TrainParams,
    n_ports::Int64,
    source_name::String,
)
    model_initializer = _instantiate_model_initializer(params, n_ports, flux = false)
    model_node = _instantiate_model_node(params, n_ports, flux = false)
    model_observation = _instantiate_model_observation(params, n_ports, flux = false)

    surr = PSIDS.SteadyStateNODE(
        name = source_name,
        initializer_structure = model_initializer,
        node_structure = model_node,
        observer_structure = model_observation,
        x_scale = params.input_normalization.x_scale,
        x_bias = params.input_normalization.x_bias,
        exogenous_scale = params.input_normalization.exogenous_scale,
        exogenous_bias = params.input_normalization.exogenous_bias,
        base_power = 100.0, #TODO - doublecheck 
        ext = Dict{String, Any}(),
    )
    display(surr)
    return surr
end

function instantiate_surrogate_flux(params::TrainParams, n_ports::Int64)
    steadystate_solver = instantiate_steadystate_solver(params.steady_state_solver)
    dynamic_solver = instantiate_solver(params.dynamic_solver)
    model_initializer = _instantiate_model_initializer(params, n_ports, flux = true)
    model_node = _instantiate_model_node(params, n_ports, flux = true)
    model_observation = _instantiate_model_observation(params, n_ports, flux = true)

    display(model_initializer)
    display(model_node)
    display(model_observation)

    dynamic_reltol = params.dynamic_solver.tols[1]
    dynamic_abstol = params.dynamic_solver.tols[2]
    dynamic_maxiters = params.dynamic_solver.maxiters
    steadystate_maxiters = params.steady_state_solver.maxiters
    steadystate_abstol = params.steady_state_solver.abstol

    return SteadyStateNeuralODE(
        model_initializer,
        model_node,
        model_observation,
        steadystate_solver,
        dynamic_solver,
        steadystate_maxiters,
        steadystate_abstol;
        abstol = dynamic_abstol,
        reltol = dynamic_reltol,
        maxiters = dynamic_maxiters,
    )
end

function _instantiate_model_initializer(params, n_ports; flux = true)
    initializer_params = params.model_initializer
    hidden_states = params.hidden_states
    type = initializer_params.type
    n_layer = initializer_params.n_layer
    width_layers = initializer_params.width_layers
    if flux == true
        activation = activation_map(initializer_params.activation)
    else
        activation = initializer_params.activation
    end
    vector_layers = []
    if type == "dense"
        if flux == true
            x_scale = params.input_normalization.x_scale
            x_bias = params.input_normalization.x_bias
            push!(vector_layers, (x) -> ((x .+ x_bias) .* x_scale))
        end
        if n_layer == 0
            if flux == true
                push!(
                    vector_layers,
                    Dense(
                        SURROGATE_SS_INPUT_DIM * n_ports,
                        hidden_states + SURROGATE_N_REFS,
                    ),
                )
            else
                push!(
                    vector_layers,
                    (
                        SURROGATE_SS_INPUT_DIM * n_ports,
                        hidden_states + SURROGATE_N_REFS,
                        true,
                        "identity",
                    ),
                )
            end
        else
            if flux == true
                push!(
                    vector_layers,
                    Dense(SURROGATE_SS_INPUT_DIM * n_ports, width_layers, activation),
                )
            else
                push!(
                    vector_layers,
                    (SURROGATE_SS_INPUT_DIM * n_ports, width_layers, true, activation),
                )
            end
            for i in 1:(n_layer - 1)
                if flux == true
                    push!(vector_layers, Dense(width_layers, width_layers, activation))
                else
                    push!(vector_layers, (width_layers, width_layers, true, activation))
                end
            end
            if flux == true
                push!(vector_layers, Dense(width_layers, hidden_states + SURROGATE_N_REFS))
            else
                push!(
                    vector_layers,
                    (width_layers, hidden_states + SURROGATE_N_REFS, true, "identity"),
                )
            end
        end
    elseif type == "OutputParams"
        @error "OutputParams layer for inititalizer not yet implemented"
    end
    if flux == true
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
        return model
    else
        return vector_layers
    end
end

function _instantiate_model_node(params, n_ports; flux = true)
    node_params = params.model_node
    hidden_states = params.hidden_states
    type = node_params.type
    n_layer = node_params.n_layer
    width_layers = node_params.width_layers
    if flux == true
        activation = activation_map(node_params.activation)
    else
        activation = node_params.activation
    end
    vector_layers = []
    if type == "dense"
        if flux == true
            push!(
                vector_layers,
                Parallel(
                    vcat,
                    (x) -> (x),
                    (x) -> (
                        (x .+ params.input_normalization.exogenous_bias) .*
                        params.input_normalization.exogenous_scale
                    ),
                    (x) -> (x),
                ),
            )
        end
        if n_layer == 0
            if flux == true
                push!(
                    vector_layers,
                    Dense(
                        hidden_states +
                        (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                        hidden_states,
                        init = NormalInitializer(),
                    ),
                )
            else
                push!(
                    vector_layers,
                    (
                        hidden_states +
                        (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                        hidden_states,
                        true,
                        "identity",
                    ),
                )
            end
        else
            if flux == true
                push!(
                    vector_layers,
                    Dense(
                        hidden_states +
                        (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                        width_layers,
                        activation,
                        init = NormalInitializer(),
                    ),
                )
            else
                push!(
                    vector_layers,
                    (
                        hidden_states +
                        (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                        width_layers,
                        true,
                        activation,
                    ),
                )
            end
            for i in 1:(n_layer - 1)
                if flux == true
                    push!(
                        vector_layers,
                        Dense(
                            width_layers,
                            width_layers,
                            activation,
                            init = NormalInitializer(),
                        ),
                    )
                else
                    push!(vector_layers, (width_layers, width_layers, true, activation))
                end
            end
            if flux == true
                push!(
                    vector_layers,
                    Dense(width_layers, hidden_states, init = NormalInitializer()),
                )
            else
                push!(vector_layers, (width_layers, hidden_states, true, "identity"))
            end
        end
    elseif type == "OutputParams"
        @error "OutputParams layer for inititalizer not yet implemented"
    end
    if flux == true
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
        return model
    else
        return vector_layers
    end
end

function _instantiate_model_observation(params, n_ports; flux = true)
    observation_params = params.model_observation
    hidden_states = params.hidden_states
    type = observation_params.type
    n_layer = observation_params.n_layer
    width_layers = observation_params.width_layers
    if flux == true
        activation = activation_map(observation_params.activation)
    else
        activation = observation_params.activation
    end
    vector_layers = []
    if type == "dense"
        if n_layer == 0
            if flux == true
                push!(
                    vector_layers,
                    Dense(hidden_states, n_ports * SURROGATE_OUTPUT_DIM),  #identity activation for output layer
                )
            else
                push!(
                    vector_layers,
                    (hidden_states, n_ports * SURROGATE_OUTPUT_DIM, true, "identity"),  #identity activation for output layer
                )
            end
        else
            if flux == true
                push!(vector_layers, Dense(hidden_states, width_layers, activation))
            else
                push!(vector_layers, (hidden_states, width_layers, true, activation))
            end
            for i in 1:(n_layer - 1)
                if flux == true
                    push!(vector_layers, Dense(width_layers, width_layers, activation))
                else
                    push!(vector_layers, (width_layers, width_layers, true, activation))
                end
            end
            if flux == true
                push!(
                    vector_layers,
                    Dense(width_layers, n_ports * SURROGATE_OUTPUT_DIM),    #identity activation for output layer
                )
            else
                push!(
                    vector_layers,
                    (width_layers, n_ports * SURROGATE_OUTPUT_DIM, true, "identity"),
                )
            end
        end
    elseif type == "OutputParams"
        @error "OutputParams layer for inititalizer not yet implemented"
    end
    if flux == true
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
        return model
    else
        return vector_layers
    end
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
    train_dataset::Array{PSIDS.SurrogateDataset},
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
        train_dataset,
        exs,
        train_details,
        params,
    )
end

function _outer_loss_function(
    θ,
    fault_timespan_index::Tuple{Int64, Int64},
    surrogate::SteadyStateNeuralODE,
    train_dataset::Array{PSIDS.SurrogateDataset},
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
    powerflow = train_dataset[fault_index].powerflow
    tsteps = train_dataset[fault_index].tsteps
    groundtruth_current = train_dataset[fault_index].groundtruth_current
    index_subset = _find_subset(tsteps, train_details[timespan_index])
    t_subset = tsteps[index_subset]
    groundtruth_subset = groundtruth_current[:, index_subset]
    surrogate_solution = surrogate(ex, powerflow, t_subset, θ)
    #=          p1 = Plots.plot(tsteps, groundtruth_current[1,:])
            Plots.plot!(p1, surrogate_solution.t_series, surrogate_solution.i_series[1,:])
            p2 = Plots.plot(tsteps, groundtruth_current[2,:])
            Plots.plot!(p2, surrogate_solution.t_series, surrogate_solution.i_series[2,:])
            display(Plots.plot(p1,p2)) =#
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

function instantiate_cb!(
    output,
    params,
    validation_dataset,
    sys_validation,
    connecting_branches,
    surrogate,
)
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
        params,
        print_loss,
        validation_dataset,
        sys_validation,
        connecting_branches,
        surrogate,
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
    params,
    print_loss,
    validation_dataset,
    sys_validation,
    connecting_branches,
    surrogate,
)
    lb_loss = params.lb_loss
    exportmode_skip = params.output_mode_skip
    train_time_limit_seconds = params.train_time_limit_seconds
    validation_loss_every_n = params.validation_loss_every_n

    push!(output["loss"], (lA, lB, lC, l))
    output["total_iterations"] += 1
    if mod(output["total_iterations"], exportmode_skip) == 0
        push!(output["predictions"], ([p], surrogate_solution, fault_index))
        push!(output["recorded_iterations"], output["total_iterations"])
    end
    #=     p1 = Plots.plot(surrogate_solution.t_series, surrogate_solution.i_series[1, :])
        p2 = Plots.plot(surrogate_solution.t_series, surrogate_solution.i_series[2, :])
        display(Plots.plot(p1, p2)) =#
    if (print_loss)
        #= 
        println(l)
        println(p[1:35])
        println(p[36:100])
        println(p[101:112])
                @info l, lA, lB, lC
                @warn surrogate_solution.r0_pred
                @warn surrogate_solution.r0
                @warn p  =#
    end
    if mod(output["total_iterations"], validation_loss_every_n) == 0
        validation_loss = evaluate_loss(
            sys_validation,
            p,
            validation_dataset,
            params.validation_data,
            connecting_branches,
            surrogate,
        )
    end
    if (l < lb_loss) || (time() > train_time_limit_seconds)
        return true
    else
        return false
    end
end
