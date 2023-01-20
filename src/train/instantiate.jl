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

function instantiate_optimizer(opt)
    if opt.algorithm == "Adam"
        return optimizer_map(opt.algorithm)(opt.η)
    elseif opt.algorithm == "Bfgs"
        return optimizer_map(opt.algorithm)(initial_stepnorm = opt.initial_stepnorm)
    elseif opt.algorithm == "LBfgs"
        return optimizer_map(opt.algorithm)()
    else
        @error "invalid algorithm provided"
    end
end

function add_surrogate_psid!(
    sys::PSY.System,
    model_params::SteadyStateNODEObsParams,
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
)
    n_ports = model_params.n_ports
    scaling_extrema = calculate_scaling_extrema(train_dataset)
    model_initializer =
        _instantiate_model_initializer(model_params, n_ports, scaling_extrema, flux = false)     #scaling_extrema not used in PSID NNs
    model_dynamic =
        _instantiate_model_dynamic(model_params, n_ports, scaling_extrema, flux = false)   #scaling_extrema not used in PSID NNs
    model_observation =
        _instantiate_model_observation(model_params, n_ports, scaling_extrema, flux = false)     #scaling_extrema not used in PSID NNs    

    source = PSY.get_component(PSY.Source, sys, model_params.name) #Note: hardcoded for single port surrogate 
    #source_name = source_names[1]

    surr = PSIDS.SteadyStateNODEObs(
        name = model_params.name,
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
    display(surr)
    @info "Surrogate: $(surr)\n"
    PSY.add_component!(sys, surr, source)
    return
end

function add_surrogate_psid!(
    sys::PSY.System,
    model_params::SteadyStateNODEParams,
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
)
    n_ports = model_params.n_ports
    scaling_extrema = calculate_scaling_extrema(train_dataset)
    model_initializer =
        _instantiate_model_initializer(model_params, n_ports, scaling_extrema, flux = false)     #scaling_extrema not used in PSID NNs
    model_dynamic =
        _instantiate_model_dynamic(model_params, n_ports, scaling_extrema, flux = false)   #scaling_extrema not used in PSID NNs

    source = PSY.get_component(PSY.Source, sys, model_params.name) #Note: hardcoded for single port surrogate 

    surr = PSIDS.SteadyStateNODE(
        name = model_params.name,
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
    display(surr)
    @info "Surrogate: $(surr)\n"
    PSY.add_component!(sys, surr, source)
    return
end

function add_surrogate_psid!(
    sys::PSY.System,
    model_params::ClassicGenParams,
    ::Vector{PSIDS.SteadyStateNODEData},   #Won't be used in this dispatch 
)
    source = PSY.get_component(PSY.Source, sys, model_params.name) #Note: hardcoded for single port surrogate 
    P_ref = PSY.get_active_power(source)
    Q_ref = PSY.get_reactive_power(source)
    b = PSY.get_bus(source)
    PSY.remove_component!(sys, source)
    static_injector = PSY.ThermalStandard(
        name = model_params.name,
        available = true,
        status = true,
        bus = b,
        active_power = P_ref,
        reactive_power = 0.0,
        rating = 2.0,
        active_power_limits = (min = 0.0, max = 2.0),
        reactive_power_limits = (min = -1.5, max = 1.5),
        time_limits = nothing,
        ramp_limits = nothing,
        operation_cost = PSY.ThreePartCost((0.0, 4000.0), 0.0, 4.0, 2.0),   #unused 
        base_power = 100.0,
    )
    PSY.add_component!(sys, static_injector)
    dynamic_injector = PSY.DynamicGenerator(
        name = model_params.name,
        ω_ref = 1.0,
        machine = PSY.BaseMachine(0.0, 0.0, 0.0),
        shaft = PSY.SingleMass(0.0, 0.0),
        avr = PSY.AVRFixed(0.0),
        prime_mover = PSY.TGFixed(1.0),
        pss = PSY.PSSFixed(0.0),
    )
    PSY.add_component!(sys, dynamic_injector, static_injector)
end

function instantiate_surrogate_flux(
    params::TrainParams,
    model_params::Union{SteadyStateNODEParams, SteadyStateNODEObsParams},
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
)
    n_ports = model_params.n_ports
    scaling_extrema = calculate_scaling_extrema(train_dataset)
    steadystate_solver = instantiate_steadystate_solver(params.steady_state_solver)
    dynamic_solver = instantiate_solver(params.dynamic_solver)
    model_initializer =
        _instantiate_model_initializer(model_params, n_ports, scaling_extrema, flux = true)
    model_dynamic =
        _instantiate_model_dynamic(model_params, n_ports, scaling_extrema, flux = true)
    model_observation =
        _instantiate_model_observation(model_params, n_ports, scaling_extrema, flux = true)

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
    steadystate_abstol = params.steady_state_solver.abstol

    return SteadyStateNeuralODE(
        model_initializer,
        model_dynamic,
        model_observation,
        steadystate_solver,
        dynamic_solver,
        steadystate_abstol;
        abstol = dynamic_abstol,
        reltol = dynamic_reltol,
        maxiters = dynamic_maxiters,
    )
end

function instantiate_surrogate_flux(
    params::TrainParams,
    model_params::ClassicGenParams,
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
)
    steadystate_solver = instantiate_steadystate_solver(params.steady_state_solver)
    dynamic_solver = instantiate_solver(params.dynamic_solver)
    dynamic_reltol = params.dynamic_solver.reltol
    dynamic_abstol = params.dynamic_solver.abstol
    dynamic_maxiters = params.dynamic_solver.maxiters
    steadystate_abstol = params.steady_state_solver.abstol

    return ClassicGen(
        steadystate_solver,
        dynamic_solver,
        steadystate_abstol;
        abstol = dynamic_abstol,
        reltol = dynamic_reltol,
        maxiters = dynamic_maxiters,
    )
end

function _instantiate_model_initializer(m, n_ports, scaling_extrema; flux = true)
    hidden_states = m.dynamic_hidden_states
    type = m.initializer_layer_type
    n_layer = m.initializer_n_layer
    width_layers = m.initializer_width_layers
    input_min = scaling_extrema["input_min"]
    input_max = scaling_extrema["input_max"]
    target_min = scaling_extrema["target_min"]
    target_max = scaling_extrema["target_max"]
    if flux == true
        activation = activation_map(m.initializer_activation)
    else
        activation = m.initializer_activation
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

function _instantiate_model_dynamic(m, n_ports, scaling_extrema; flux = true)
    hidden_states = m.dynamic_hidden_states
    type = m.dynamic_layer_type
    n_layer = m.dynamic_n_layer
    width_layers = m.dynamic_width_layers
    σ2_initialization = m.dynamic_σ2_initialization
    input_min = scaling_extrema["input_min"]
    input_max = scaling_extrema["input_max"]
    if flux == true
        activation = activation_map(m.dynamic_activation)
    else
        activation = m.dynamic_activation
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

function _instantiate_model_observation(m, n_ports, scaling_extrema; flux = true)
    target_min = scaling_extrema["target_min"]
    target_max = scaling_extrema["target_max"]
    vector_layers = []
    if typeof(m) == SteadyStateNODEParams
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
    elseif typeof(m) == SteadyStateNODEObsParams
        if flux == true
            activation = activation_map(m.observation_activation)
        else
            activation = m.observation_activation
        end
        hidden_states = m.dynamic_hidden_states
        type = m.observation_layer_type
        n_layer = m.observation_n_layer
        width_layers = m.observation_width_layers
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
    surrogate_solution::SteadyStateNeuralODE_solution,
    real_current_subset,
    imag_current_subset,
    params,
    opt_ix,
)
    ground_truth_subset = vcat(real_current_subset, imag_current_subset)
    rmse_weight = params.optimizer[opt_ix].loss_function.type_weights.rmse
    mae_weight = params.optimizer[opt_ix].loss_function.type_weights.mae
    initialization_weight =
        params.optimizer[opt_ix].loss_function.component_weights.initialization_weight
    dynamic_weight = params.optimizer[opt_ix].loss_function.component_weights.dynamic_weight
    residual_penalty =
        params.optimizer[opt_ix].loss_function.component_weights.residual_penalty
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

function _inner_loss_function(
    surrogate_solution::PhysicalModel_solution,
    real_current_subset,
    imag_current_subset,
    params,
    opt_ix,
)
    ground_truth_subset = vcat(real_current_subset, imag_current_subset)
    rmse_weight = params.optimizer[opt_ix].loss_function.type_weights.rmse
    mae_weight = params.optimizer[opt_ix].loss_function.type_weights.mae
    initialization_weight =
        params.optimizer[opt_ix].loss_function.component_weights.initialization_weight
    dynamic_weight = params.optimizer[opt_ix].loss_function.component_weights.dynamic_weight
    residual_penalty =
        params.optimizer[opt_ix].loss_function.component_weights.residual_penalty
    i_series = surrogate_solution.i_series
    res = surrogate_solution.res
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
    return 0.0, loss_dynamic
end

function instantiate_outer_loss_function(
    surrogate::Union{SteadyStateNeuralODE, ClassicGen},
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
    opt_ix::Int64,
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
        opt_ix,
    )
end

function _outer_loss_function(
    p_train,
    vector_fault_timespan_index::Vector{Tuple{Int64, Int64}},
    surrogate::Union{SteadyStateNeuralODE, ClassicGen},
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
    opt_ix::Int64,
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
        if params.dynamic_solver.force_tstops == true
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
        surrogate_solution = surrogate(
            V,
            [vr0, vi0],
            [ir0, ii0],
            tsteps_subset,
            tstops,     #if entry in tstops is outside of tspan, will be ignored.
            p_fixed,
            p_train,
            p_map,
        )
        loss_i, loss_d = _inner_loss_function(
            surrogate_solution,
            real_current_subset,
            imag_current_subset,
            params,
            opt_ix,
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
    data_collection_location,
    surrogate,
    p_fixed,
    p_map,
    opt_ix,
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
            data_collection_location,
            surrogate,
            p_fixed,
            p_map,
            opt_ix,
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
    data_collection_location,
    surrogate,
    p_fixed,
    p_map,
    opt_ix,
)
    lb_loss = params.optimizer[opt_ix].lb_loss
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
            data_collection_location,
            params.model_params,
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
