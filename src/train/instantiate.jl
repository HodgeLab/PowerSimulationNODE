function optimizer_map(key)
    d = Dict(
        "Adam" => OptimizationOptimisers.Optimisers.ADAM,
        "Bfgs" => OptimizationOptimJL.Optim.BFGS,
        "LBfgs" => OptimizationOptimJL.Optim.LBFGS,
    )
    return d[key]
end

function NormalInitializer(μ = 0.0f0; σ² = 0.05f0)
    return (dims...) -> randn(Float32, dims...) .* Float32(σ²) .+ Float32(μ)
end

function steadystate_solver_map(solver, tols)
    d = Dict(
        "Rodas4" => SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.Rodas4()),
        "Rodas5" => SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.Rodas5()),
        #"TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => SteadyStateDiffEq.DynamicSS(OrdinaryDiffEq.Tsit5()),
        "SSRootfind" => SteadyStateDiffEq.SSRootfind(),
    )
    return d[solver]
end

function solver_map(key)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "Rodas5" => OrdinaryDiffEq.Rodas5,
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
        "ForwardDiff" => Optimization.AutoForwardDiff,
        "Zygote" => Optimization.AutoZygote,
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
    end
end

function instantiate_optimizer_adjust(inputs)
    if inputs.optimizer.adjust == "Bfgs"
        return optimizer_map(inputs.optimizer.adjust)(
            initial_stepnorm = inputs.optimizer.adjust_initial_stepnorm,
        )
    elseif inputs.optimizer.adjust == "LBfgs"
        return optimizer_map(inputs.optimizer.adjust)()
    end
end

function instantiate_surrogate_psid(
    params::TrainParams,
    n_ports::Int64,
    scaling_extrema::Dict{String, Vector{Float64}},
    source_name::String,
)
    model_initializer =
        _instantiate_model_initializer(params, n_ports, scaling_extrema, flux = false)     #scaling_extrema not used in PSID NNs
    model_dynamic =
        _instantiate_model_dynamic(params, n_ports, scaling_extrema, flux = false)   #scaling_extrema not used in PSID NNs

    if params.model_observation.type == "dense"   #If the type is "dense", instantiate SteadyStateNODEObs
        model_observation =
            _instantiate_model_observation(params, n_ports, scaling_extrema, flux = false)     #scaling_extrema not used in PSID NNs    
        surr = PSIDS.SteadyStateNODEObs(
            name = source_name,
            initializer_structure = model_initializer,
            node_structure = model_dynamic,
            observer_structure = model_observation,
            input_min = scaling_extrema["input_min"],
            input_max = scaling_extrema["input_max"],
            input_lims = (NN_INPUT_LIMITS.min, NN_INPUT_LIMITS.max),
            target_min = scaling_extrema["target_min"],
            target_max = scaling_extrema["target_max"],
            target_lims = (NN_TARGET_LIMITS.min, NN_TARGET_LIMITS.max),
            base_power = 100.0,
            ext = Dict{String, Any}(),
        )
    elseif params.model_observation.type == "DirectObservation" #If the type is "DirectObservation", instantiate SteadyStateNODE
        surr = PSIDS.SteadyStateNODE(
            name = source_name,
            initializer_structure = model_initializer,
            node_structure = model_dynamic,
            input_min = scaling_extrema["input_min"],
            input_max = scaling_extrema["input_max"],
            input_lims = (NN_INPUT_LIMITS.min, NN_INPUT_LIMITS.max),
            target_min = scaling_extrema["target_min"],
            target_max = scaling_extrema["target_max"],
            target_lims = (NN_TARGET_LIMITS.min, NN_TARGET_LIMITS.max),
            base_power = 100.0,
            ext = Dict{String, Any}(),
        )
    else
        @error "Invalid parameter for observation function"
    end
    display(surr)
    @info "Surrogate: $(surr)\n"
    return surr
end

function instantiate_surrogate_flux(
    params::TrainParams,
    n_ports::Int64,
    scaling_extrema::Dict{String, Vector{Float64}},
)
    steadystate_solver = instantiate_steadystate_solver(params.steady_state_solver)
    dynamic_solver = instantiate_solver(params.dynamic_solver)
    model_initializer =
        _instantiate_model_initializer(params, n_ports, scaling_extrema, flux = true)
    model_dynamic =
        _instantiate_model_dynamic(params, n_ports, scaling_extrema, flux = true)
    model_observation =
        _instantiate_model_observation(params, n_ports, scaling_extrema, flux = true)

    display(model_initializer)
    display(model_dynamic)
    display(model_observation)
    @info "Iniitalizer structure: $(model_initializer)\n"
    @info "number of parameters: $(length(Flux.destructure(model_initializer)[1]))\n"
    @info "NODE structure: $(model_dynamic)\n"
    @info "number of parameters: $(length(Flux.destructure(model_dynamic)[1]))\n"
    @info "Observation structure: $(model_observation)\n"
    @info "number of parameters: $(length(Flux.destructure(model_observation)[1]))\n"
    dynamic_reltol = params.dynamic_solver.reltol
    dynamic_abstol = params.dynamic_solver.abstol
    dynamic_maxiters = params.dynamic_solver.maxiters
    steadystate_maxiters = params.steady_state_solver.maxiters
    steadystate_abstol = params.steady_state_solver.abstol

    return SteadyStateNeuralODE(
        model_initializer,
        model_dynamic,
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

function _instantiate_model_initializer(params, n_ports, scaling_extrema; flux = true)
    initializer_params = params.model_initializer
    hidden_states = params.model_dynamic.hidden_states
    type = initializer_params.type
    n_layer = initializer_params.n_layer
    width_layers = initializer_params.width_layers
    input_min = scaling_extrema["input_min"]
    input_max = scaling_extrema["input_max"]
    target_min = scaling_extrema["target_min"]
    target_max = scaling_extrema["target_max"]
    if flux == true
        activation = activation_map(initializer_params.activation)
    else
        activation = initializer_params.activation
    end
    vector_layers = []
    if type == "dense"
        if flux == true
            push!(
                vector_layers,
                Parallel(
                    vcat,
                    (x) -> (
                        (x - input_min[2]) / (input_max[2] - input_min[2]) *
                        (NN_INPUT_LIMITS[2] - NN_INPUT_LIMITS[1]) + NN_INPUT_LIMITS[1]
                    ),    #Only pass Vq (Vd=0 by definition)
                    (x) -> (
                        (x .- target_min) ./ (target_max .- target_min) .*
                        (NN_TARGET_LIMITS[2] .- NN_TARGET_LIMITS[1]) .+
                        NN_TARGET_LIMITS[1]
                    ),        #same as PSIDS.min_max_normalization
                ),
            )
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

function _instantiate_model_dynamic(params, n_ports, scaling_extrema; flux = true)
    node_params = params.model_dynamic
    hidden_states = node_params.hidden_states
    type = node_params.type
    n_layer = node_params.n_layer
    width_layers = node_params.width_layers
    σ2_initialization = node_params.σ2_initialization
    input_min = scaling_extrema["input_min"]
    input_max = scaling_extrema["input_max"]
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
                    (x) -> x,
                    (x) -> (
                        (x .- input_min) ./ (input_max .- input_min) .*
                        (NN_INPUT_LIMITS[2] .- NN_INPUT_LIMITS[1]) .+
                        NN_INPUT_LIMITS[1]
                    ),
                    (x) -> x,
                ),
            )
        end
        if n_layer == 0
            if flux == true
                if σ2_initialization == 0.0
                    push!(
                        vector_layers,
                        Dense(
                            hidden_states +
                            (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                            hidden_states,
                        ),
                    )
                else
                    push!(
                        vector_layers,
                        Dense(
                            hidden_states +
                            (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                            hidden_states,
                            init = NormalInitializer(σ² = σ2_initialization),
                        ),
                    )
                end
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
                if σ2_initialization == 0.0
                    push!(
                        vector_layers,
                        Dense(
                            hidden_states +
                            (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                            width_layers,
                            activation,
                        ),
                    )
                else
                    push!(
                        vector_layers,
                        Dense(
                            hidden_states +
                            (SURROGATE_EXOGENOUS_INPUT_DIM + SURROGATE_N_REFS) * n_ports,
                            width_layers,
                            activation,
                            init = NormalInitializer(σ² = σ2_initialization),
                        ),
                    )
                end
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
                    if σ2_initialization == 0.0
                        push!(vector_layers, Dense(width_layers, width_layers, activation))
                    else
                        push!(
                            vector_layers,
                            Dense(
                                width_layers,
                                width_layers,
                                activation,
                                init = NormalInitializer(σ² = σ2_initialization),
                            ),
                        )
                    end
                else
                    push!(vector_layers, (width_layers, width_layers, true, activation))
                end
            end
            if flux == true
                if σ2_initialization == 0.0
                    push!(vector_layers, Dense(width_layers, hidden_states))
                else
                    push!(
                        vector_layers,
                        Dense(
                            width_layers,
                            hidden_states,
                            init = NormalInitializer(σ² = σ2_initialization),
                        ),
                    )
                end
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

function _instantiate_model_observation(params, n_ports, scaling_extrema; flux = true)
    observation_params = params.model_observation
    hidden_states = params.model_dynamic.hidden_states
    type = observation_params.type
    n_layer = observation_params.n_layer
    width_layers = observation_params.width_layers
    target_min = scaling_extrema["target_min"]
    target_max = scaling_extrema["target_max"]
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
                push!(
                    vector_layers,
                    (x) -> (
                        (x .- NN_TARGET_LIMITS[1]) .* (target_max .- target_min) ./
                        (NN_TARGET_LIMITS[2] .- NN_TARGET_LIMITS[1]) .+ target_min
                    ),
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
                push!(
                    vector_layers,
                    (x) -> (
                        (x .- NN_TARGET_LIMITS[1]) .* (target_max .- target_min) ./
                        (NN_TARGET_LIMITS[2] .- NN_TARGET_LIMITS[1]) .+ target_min
                    ),
                )
            else
                push!(
                    vector_layers,
                    (width_layers, n_ports * SURROGATE_OUTPUT_DIM, true, "identity"),
                )
            end
        end
    elseif type == "DirectObservation"
        if flux == true
            push!(vector_layers, (x) -> (x[1:(n_ports * SURROGATE_OUTPUT_DIM), :]))
            push!(
                vector_layers,
                (x) -> (
                    (x .- NN_TARGET_LIMITS[1]) .* (target_max .- target_min) ./
                    (NN_TARGET_LIMITS[2] .- NN_TARGET_LIMITS[1]) .+ target_min
                ),
            )
        else
            @error "DirectObservation incompatible with instantiating observation for PSID"
            @assert false
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

function _inner_loss_function(
    surrogate_solution,
    real_current_subset,
    imag_current_subset,
    params,
)
    ground_truth_subset = vcat(real_current_subset, imag_current_subset)
    rmse_weight = params.loss_function.type_weights.rmse
    mae_weight = params.loss_function.type_weights.mae
    initialization_weight = params.loss_function.component_weights.initialization_weight
    dynamic_weight = params.loss_function.component_weights.dynamic_weight
    residual_penalty = params.loss_function.component_weights.residual_penalty
    r0_pred = surrogate_solution.r0_pred
    r0 = surrogate_solution.r0
    i_series = surrogate_solution.i_series
    res = surrogate_solution.res
    loss_initialization =
        initialization_weight *
        (mae_weight * mae(r0_pred, r0) + rmse_weight * sqrt(mse(r0_pred, r0)))
    #Note: A loss of 0.0 makes the NLsolve equations non-finite during training. Instead, set the loss to the tolerance of the NLsolve. 
    #if loss_initialization < params.steady_state_solver.abstol
    if loss_initialization == 0.0
        loss_initialization = params.steady_state_solver.abstol
    end
    if size(ground_truth_subset) == size(i_series)
        loss_dynamic =
            dynamic_weight * (
                mae_weight * mae(ground_truth_subset, i_series) +
                rmse_weight * sqrt(mse(ground_truth_subset, i_series))
            )
    else
        loss_dynamic =
            residual_penalty * (
                mae_weight * mae(res, zeros(length(res))) +
                rmse_weight * sqrt(mse(res, zeros(length(res))))
            )
    end
    return loss_initialization, loss_dynamic
end

function instantiate_outer_loss_function(
    surrogate::SteadyStateNeuralODE,
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
    exogenous_input_functions,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    p_fixed::Vector{Float32},
    p_map::Vector{Int64},
    params::TrainParams,
)
    return (p_train, vector_fault_timespan_index) -> _outer_loss_function(
        p_train,
        vector_fault_timespan_index,
        surrogate,
        train_dataset,
        exogenous_input_functions,
        train_details,
        p_fixed,
        p_map,
        params,
    )
end

function _outer_loss_function(
    p_train,
    vector_fault_timespan_index::Vector{Tuple{Int64, Int64}},
    surrogate::SteadyStateNeuralODE,
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
    exogenous_input_functions,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    p_fixed::Vector{Float32},
    p_map::Vector{Int64},
    params::TrainParams,
)
    vector_fault_timespan_index
    surrogate_solution = 0.0    #Only return the surrogate_solution from the last fault of the iteration (cannot mutate arrays with Zygote)
    loss_initialization = 0.0
    loss_dynamic = 0.0
    for (ix, fault_timespan_index) in enumerate(vector_fault_timespan_index)
        fault_index = fault_timespan_index[1]
        timespan_index = fault_timespan_index[2]
        V = exogenous_input_functions[fault_index]
        #powerflow = train_dataset[fault_index].powerflow
        vr0 = train_dataset[fault_index].surrogate_real_voltage[1]
        vi0 = train_dataset[fault_index].surrogate_imag_voltage[1]
        ir0 = train_dataset[fault_index].real_current[1]
        ii0 = train_dataset[fault_index].imag_current[1]

        tsteps = train_dataset[fault_index].tsteps
        if params.force_tstops == true
            tstops = train_dataset[fault_index].tstops
        else
            tstops = []
        end
        index_subset = _find_subset_batching(tsteps, train_details[timespan_index])
        real_current = train_dataset[fault_index].real_current
        real_current_subset = real_current[:, index_subset]
        imag_current = train_dataset[fault_index].imag_current
        imag_current_subset = imag_current[:, index_subset]
        tsteps_subset = tsteps[index_subset]
        #tstops_subset = tstops[index_subset]   #TODO - does it matter if we pass the entire tstops (outside of the range?)
        surrogate_solution = surrogate(
            V,
            [vr0, vi0],
            [ir0, ii0],
            tsteps_subset,
            tstops,
            p_fixed,
            p_train,
            p_map,
        )
        loss_i, loss_d = _inner_loss_function(
            surrogate_solution,
            real_current_subset,
            imag_current_subset,
            params,
        )
        loss_initialization += loss_i
        loss_dynamic += loss_d
    end
    return loss_initialization + loss_dynamic,
    loss_initialization,
    loss_dynamic,
    surrogate_solution,    #Note- this is the last surrogate_solution only (if there were multiple)
    vector_fault_timespan_index
end

function _find_subset_batching(tsteps, train_details)
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
    p_fixed,
    p_map,
    θ_ranges,
)
    if Sys.iswindows() || Sys.isapple()
        print_loss = true
    else
        print_loss = false
    end

    return (p_train, l, l_initialization, l_dynamic, surrogate_solution, fault_index) ->
        _cb!(
            p_train,
            l,
            l_initialization,
            l_dynamic,
            surrogate_solution,
            fault_index,
            output,
            params,
            print_loss,
            validation_dataset,
            sys_validation,
            connecting_branches,
            surrogate,
            p_fixed,
            p_map,
            θ_ranges,
        )
end

function _cb!(
    p_train,
    l,
    l_initialization,
    l_dynamic,
    surrogate_solution,
    fault_index,
    output,
    params,
    print_loss,
    validation_dataset,
    sys_validation,
    connecting_branches,
    surrogate,
    p_fixed,
    p_map,
    θ_ranges,
)
    lb_loss = params.lb_loss
    exportmode_skip = params.output_mode_skip
    train_time_limit_seconds = params.train_time_limit_seconds
    validation_loss_every_n = params.validation_loss_every_n

    push!(output["loss"], (l_initialization, l_dynamic, l, surrogate_solution.converged))
    if mod(output["total_iterations"], exportmode_skip) == 0 ||
       output["total_iterations"] == 1
        push!(
            output["predictions"],
            ([vcat(p_fixed, p_train)[p_map]], surrogate_solution, fault_index),
        )
        push!(output["recorded_iterations"], output["total_iterations"])
    end
    output["total_iterations"] += 1
    #=     p1 = Plots.plot(surrogate_solution.t_series, surrogate_solution.i_series[1, :])
        p2 = Plots.plot(surrogate_solution.t_series, surrogate_solution.i_series[2, :])
        display(Plots.plot(p1, p2)) =#
    if (print_loss)
        println(
            "total loss: ",
            l,
            "\t init loss: ",
            l_initialization,
            "\t dynamic loss: ",
            l_dynamic,
            "\t fault/timespan index: ",
            fault_index,
        )
    end
    if mod(output["total_iterations"], validation_loss_every_n) == 0
        validation_loss = evaluate_loss(
            sys_validation,
            vcat(p_fixed, p_train)[p_map],
            validation_dataset,
            params.validation_data,
            connecting_branches,
            θ_ranges,
        )

        push!(
            output["validation_loss"],
            (
                validation_loss["mae_ir"],
                validation_loss["max_error_ir"],
                validation_loss["mae_ii"],
                validation_loss["max_error_ii"],
            ),
        )
        ir_mean = Statistics.mean(validation_loss["mae_ir"])
        ii_mean = Statistics.mean(validation_loss["mae_ii"])
        if (0.0 < ((ir_mean + ii_mean) / 2) < lb_loss)  # validation loss assigned 0.0 when not stable
            @warn "Training stopping condition met: loss validation set is below defined limit"
            return true
        end
    end
    if (floor(time()) > train_time_limit_seconds)
        @warn "Training stopping condition met: time limit is up"
        return true
    else
        return false
    end
end
