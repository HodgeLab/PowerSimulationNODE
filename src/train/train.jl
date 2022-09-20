
function _build_exogenous_input_functions(
    train_data_params::NamedTuple{
        (:id, :operating_points, :perturbations, :params, :system),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
            String,
        },
    },
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
)
    exs = []
    n_perturbations = length(train_data_params.perturbations)
    n_operating_points = length(train_data_params.operating_points)
    @assert n_perturbations * n_operating_points == length(train_dataset)
    for ix_o in 1:n_operating_points   #Note! Nested loop in same order as generate_surrogate_data from PSIDS
        for (ix_p, p) in enumerate(train_data_params.perturbations)
            ix = (ix_o - 1) * n_perturbations + ix_p
            train_data = train_dataset[ix]
            if train_data.stable == true    #only build exogenous inputs for the stable trajectories
                ex = _surrogate_perturbation_to_function_of_time(
                    p,
                    train_data,
                    train_data_params,
                )
                push!(exs, ex)
            end
        end
    end
    return exs
end

function _surrogate_perturbation_to_function_of_time(
    perturbation::T,
    data::D,
    train_data_params,
) where {
    T <: Vector{Union{PSID.Perturbation, PSIDS.SurrogatePerturbation}},
    D <: PSIDS.SteadyStateNODEData,
}
    V_funcs = []
    if train_data_params.system == "full"
        for i in 1:size(data.opposite_real_voltage)[1]
            function Vr(t)
                ix_after = findfirst(x -> x > t, data.tsteps)
                if ix_after === nothing  #correct behavior at end of timespan
                    ix_after = length(data.tsteps)
                end
                t_before = data.tsteps[ix_after - 1]
                t_after = data.tsteps[ix_after]
                val_before = data.opposite_real_voltage[i, ix_after - 1]
                val_after = data.opposite_real_voltage[i, ix_after]
                frac = (t - t_before) / (t_after - t_before)
                val = val_before + frac * (val_after - val_before)
                return val
            end
            function Vi(t)
                ix_after = findfirst(x -> x > t, data.tsteps)
                if ix_after === nothing  #correct behavior at end of timespan
                    ix_after = length(data.tsteps)
                end
                t_before = data.tsteps[ix_after - 1]
                t_after = data.tsteps[ix_after]
                val_before = data.opposite_imag_voltage[i, ix_after - 1]
                val_after = data.opposite_imag_voltage[i, ix_after]
                frac = (t - t_before) / (t_after - t_before)
                val = val_before + frac * (val_after - val_before)
                return val
            end
            push!(V_funcs, Vr)
            push!(V_funcs, Vi)
        end

    elseif train_data_params.system == "reduced"
        for (ix, p_single) in enumerate(perturbation)
            Vr, Vi = _single_perturbation_to_function_of_time(p_single, data, ix)
            push!(V_funcs, Vr)
            push!(V_funcs, Vi)
        end
    else
        @error "invalid value"
    end
    R = data.connecting_resistance
    X = data.connecting_reactance
    RX = collect(Iterators.flatten(zip(R, X)))
    return generate_exogenous_input(V_funcs, RX)    #Vfuncs are opposite the surrogate
end

function _single_perturbation_to_function_of_time(
    single_perturbation::P,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
) where {P <: Union{PSID.Perturbation, PSIDS.SurrogatePerturbation}}
    @error "Cannot convert the given perturbation to a function of time for training"
end

function _single_perturbation_to_function_of_time(
    single_perturbation::PSIDS.PVS,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
)
    Vr0_opposite = data.opposite_real_voltage[port_ix, 1]
    Vi0_opposite = data.opposite_imag_voltage[port_ix, 1]
    Vm0_opposite = sqrt(Vr0_opposite^2 + Vi0_opposite^2)
    θ0_opposite = atan(Vi0_opposite / Vr0_opposite)

    V_bias = Vm0_opposite
    V_freqs = single_perturbation.internal_voltage_frequencies
    V_coeffs = single_perturbation.internal_voltage_coefficients
    function V(t)
        val = V_bias
        for (i, ω) in enumerate(V_freqs)
            val += V_coeffs[i][1] * sin.(ω * t)
            val += V_coeffs[i][2] * cos.(ω * t)
        end
        return val
    end
    θ_bias = θ0_opposite
    θ_freqs = single_perturbation.internal_angle_frequencies
    θ_coeffs = single_perturbation.internal_angle_coefficients
    function θ(t)
        val = θ_bias
        for (i, ω) in enumerate(θ_freqs)
            val += θ_coeffs[i][1] * sin.(ω * t)
            val += θ_coeffs[i][2] * cos.(ω * t)
        end
        return val
    end
    function Vr_func(t)
        return V(t) * cos(θ(t))
    end
    function Vi_func(t)
        return V(t) * sin(θ(t))
    end
    return (Vr_func, Vi_func)
end

function _single_perturbation_to_function_of_time(
    single_perturbation::PSIDS.VStep,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
)
    @warn "function _single_perturbation_to_function_of_time not implemented for VStep"
end

function evaluate_loss(
    sys,
    θ,
    groundtruth_dataset,
    params,
    connecting_branches,
    surrogate,
    θ_ranges,
)
    for s in PSY.get_components(PSIDS.SteadyStateNODE, sys)
        PSIDS.set_initializer_parameters!(s, θ[θ_ranges["initializer_range"]])
        PSIDS.set_node_parameters!(s, θ[θ_ranges["node_range"]])
        PSIDS.set_observer_parameters!(s, θ[θ_ranges["observation_range"]])
    end
    operating_points = params.operating_points
    perturbations = params.perturbations
    generate_data_params = params.params
    surrogate_dataset = PSIDS.generate_surrogate_data(
        sys,
        sys,
        perturbations,
        operating_points,
        PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        generate_data_params,
        dataset_aux = groundtruth_dataset,
    )
    @assert length(surrogate_dataset) == length(groundtruth_dataset)
    mae_ir = Float64[]
    max_error_ir = Float64[]
    mae_ii = Float64[]
    max_error_ii = Float64[]
    for ix in eachindex(surrogate_dataset, groundtruth_dataset)
        if groundtruth_dataset[ix].stable == true
            if surrogate_dataset[ix].stable == false
                push!(mae_ir, 1.0e15)   #Note: Cannot write Inf in Json spec, so assign large value
                push!(max_error_ir, 1.0e15)
                push!(mae_ii, 1.0e15)
                push!(max_error_ii, 1.0e15)
            elseif surrogate_dataset[ix].stable == true
                push!(
                    mae_ir,
                    mae(
                        surrogate_dataset[ix].branch_real_current,
                        groundtruth_dataset[ix].branch_real_current,
                    ),
                )
                push!(
                    max_error_ir,
                    maximum(
                        surrogate_dataset[ix].branch_real_current .-
                        groundtruth_dataset[ix].branch_real_current,
                    ),
                )
                push!(
                    mae_ii,
                    mae(
                        surrogate_dataset[ix].branch_imag_current,
                        groundtruth_dataset[ix].branch_imag_current,
                    ),
                )
                push!(
                    max_error_ii,
                    maximum(
                        surrogate_dataset[ix].branch_imag_current .-
                        groundtruth_dataset[ix].branch_imag_current,
                    ),
                )
            end
        end
    end
    dataset_loss = Dict{String, Vector{Float64}}(
        "mae_ir" => mae_ir,
        "max_error_ir" => max_error_ir,
        "mae_ii" => mae_ii,
        "max_error_ii" => max_error_ii,
    )
    return dataset_loss
end

function calculate_scaling_extrema(train_dataset)
    n_ports = size(train_dataset[1].branch_real_current)[1]

    d_current_min = fill(Inf, n_ports)
    q_current_min = fill(Inf, n_ports)
    d_voltage_min = fill(Inf, n_ports)
    q_voltage_min = fill(Inf, n_ports)

    d_current_max = fill(-Inf, n_ports)
    q_current_max = fill(-Inf, n_ports)
    d_voltage_max = fill(-Inf, n_ports)
    q_voltage_max = fill(-Inf, n_ports)

    for d in train_dataset
        if d.stable == true
            θ0 = atan(d.surrogate_imag_voltage[1] / d.surrogate_real_voltage[1])
            id_iq = PSID.ri_dq(θ0) * vcat(d.branch_real_current, d.branch_imag_current)
            vd_vq =
                PSID.ri_dq(θ0) * vcat(d.surrogate_real_voltage, d.surrogate_imag_voltage)
            id_max = maximum(id_iq[1, :])
            id_min = minimum(id_iq[1, :])
            iq_max = maximum(id_iq[2, :])
            iq_min = minimum(id_iq[2, :])
            vd_max = maximum(vd_vq[1, :])
            vd_min = minimum(vd_vq[1, :])
            vq_max = maximum(vd_vq[2, :])
            vq_min = minimum(vd_vq[2, :])
            #display(Plots.plot(Plots.plot(id_iq[1,:]), Plots.plot(id_iq[2,:]), Plots.plot(vd_vq[1,:]), Plots.plot(vd_vq[2,:])  ))
            for ix in 1:n_ports
                #D CURRENT 
                if id_max[ix] > d_current_max[ix]
                    d_current_max[ix] = id_max[ix]
                end
                if id_min[ix] < d_current_min[ix]
                    d_current_min[ix] = id_min[ix]
                end
                #Q CURRENT 
                if iq_max[ix] > q_current_max[ix]
                    q_current_max[ix] = iq_max[ix]
                end
                if iq_min[ix] < q_current_min[ix]
                    q_current_min[ix] = iq_min[ix]
                end
                #D VOLTAGE 
                if vd_max[ix] > d_voltage_max[ix]
                    d_voltage_max[ix] = vd_max[ix]
                end
                if vd_min[ix] < d_voltage_min[ix]
                    d_voltage_min[ix] = vd_min[ix]
                end
                #Q VOLTAGE 
                if vq_max[ix] > q_voltage_max[ix]
                    q_voltage_max[ix] = vq_max[ix]
                end
                if vq_min[ix] < q_voltage_min[ix]
                    q_voltage_min[ix] = vq_min[ix]
                end
            end
        end
    end
    input_min = [d_voltage_min q_voltage_min]'[:]
    target_min = [d_current_min q_current_min]'[:]

    input_max = [d_voltage_max q_voltage_max]'[:]
    target_max = [d_current_max q_current_max]'[:]
    scaling_parameters = Dict{}(
        "input_min" => input_min,
        "input_max" => input_max,
        "target_min" => target_min,
        "target_max" => target_max,
    )
    return scaling_parameters
end
"""
    train(params::TrainParams)

Executes training according to params. Assumes the existence of the necessary input files. 

"""
function train(params::TrainParams)
    params.train_time_limit_seconds += floor(time())
    Random.seed!(params.rng_seed)

    #READ DATASETS 
    train_dataset = Serialization.deserialize(params.train_data_path)
    validation_dataset = Serialization.deserialize(params.validation_data_path)
    test_dataset = Serialization.deserialize(params.test_data_path)
    connecting_branches = Serialization.deserialize(params.connecting_branch_names_path)
    @info "Length of possible training conditions (number of fault/operating point combinations):",
    length(train_dataset)
    @info "Length of actual training dataset (stable conditions):",
    length(filter(x -> x.stable == true, train_dataset))
    @info "Length of possible validation conditions (number of fault/operating point combinations):",
    length(validation_dataset)
    @info "Length of actual validation dataset (stable conditions):",
    length(filter(x -> x.stable == true, validation_dataset))
    @info "Length of possible test conditions (number of fault/operating point combinations):",
    length(test_dataset)
    @info "Length of actual test dataset (stable conditions):",
    length(filter(x -> x.stable == true, test_dataset))

    n_ports = length(connecting_branches)
    @info "Surrogate contains $n_ports ports"

    scaling_extrema = calculate_scaling_extrema(train_dataset)

    #READ VALIDATION SYSTEM AND ADD SURROGATE COMPONENT WITH STRUCTURE BASED ON PARAMS
    sys_validation = node_load_system(params.surrogate_system_path)
    sources = collect(
        PSY.get_components(PSY.Source, sys_validation, x -> PSY.get_name(x) !== "InfBus"),
    )
    (length(sources) > 1) &&
        @error "Surrogate with multiple input/output ports not yet supported"
    psid_surrogate = instantiate_surrogate_psid(
        params,
        n_ports,
        scaling_extrema,
        PSY.get_name(sources[1]),
    )
    PSY.add_component!(sys_validation, psid_surrogate, sources[1])
    display(sys_validation)

    #INSTANTIATE 
    surrogate = instantiate_surrogate_flux(params, n_ports, scaling_extrema)
    optimizer = instantiate_optimizer(params)
    !(params.optimizer.adjust == "nothing") &&
        (optimizer_adjust = instantiate_optimizer_adjust(params))

    p_nn_init, _ = Flux.destructure(surrogate)
    n_parameters = length(p_nn_init)
    output = _initialize_training_output_dict()
    θ_ranges = Dict{String, UnitRange{Int64}}(
        "initializer_range" => 1:(surrogate.len),
        "node_range" => (surrogate.len + 1):(surrogate.len + surrogate.len2),
        "observation_range" => (surrogate.len + surrogate.len2 + 1):n_parameters,
    )
    output["θ_ranges"] = θ_ranges
    @info "Surrogate has $n_parameters parameters"
    res = nothing
    output["train_id"] = params.train_id
    output["n_params_surrogate"] = n_parameters
    exs = _build_exogenous_input_functions(params.train_data, train_dataset)    #can build ex from the components in params.train_data or from the dataset values by interpolating...
    #want to test how this impacts the speed of a single train iteration (the interpolation)
    #@warn exs[1](1.0, [1, 1])      #evaluate exs before going to the training

    train_details = params.curriculum_timespans
    fault_indices = collect(1:length(train_dataset))
    timespan_indices = collect(1:length(params.curriculum_timespans))
    @assert length(train_dataset) == length(exs)
    train_groups =
        _generate_training_groups(fault_indices, timespan_indices, params.curriculum)
    n_trains = length(train_groups)
    n_samples = length(filter(x -> x.stable == true, train_dataset))
    per_solve_max_epochs = _calculate_per_solve_max_epochs(params, n_trains, n_samples)
    @info "Curriculum pairings (fault_index, timespan_index)", train_groups
    @info "\n # of trainings: $n_trains \n # of epochs per training: $per_solve_max_epochs \n # of samples per epoch:  $n_samples"
    if isempty(params.p_start)
        θ = p_nn_init
    else
        θ = params.p_start
    end
    try # -TODO uncomment 
        total_time = @elapsed begin
            for group in train_groups
                res, output = _train(
                    θ,
                    surrogate,
                    θ_ranges,
                    connecting_branches,
                    train_dataset,
                    validation_dataset,
                    sys_validation,
                    exs,
                    train_details,
                    params,
                    optimizer,
                    group,
                    per_solve_max_epochs,
                    output,
                )
                θ = res.u
            end
        end
        output["total_time"] = total_time

        output["final_loss"] = evaluate_loss(
            sys_validation,
            θ,
            validation_dataset,
            params.validation_data,
            connecting_branches,
            surrogate,
            θ_ranges,
        )

        if params.force_gc == true
            GC.gc()     #Run garbage collector manually before file write.
            @warn "FORCE GC!"
        end
        _capture_output(output, params.output_data_path, params.train_id)
        return true, θ
    catch e
        @warn e
        return false, θ
    end
end

function _train(
    θ::Vector{Float32},
    surrogate::SteadyStateNeuralODE,
    θ_ranges::Dict{String, UnitRange{Int64}},
    connecting_branches::Vector{Tuple{String, Symbol}},
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
    validation_dataset::Vector{PSIDS.SteadyStateNODEData},
    sys_validation::PSY.System,
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
    optimizer::Union{OptimizationOptimisers.Optimisers.Adam, Optim.AbstractOptimizer},
    group::Vector{Tuple{Int64, Int64}},
    per_solve_max_epochs::Int,
    output::Dict{String, Any},
)
    train_loader =
        Flux.Data.DataLoader(group; batchsize = 1, shuffle = true, partial = true)
    outer_loss_function = instantiate_outer_loss_function(
        surrogate,
        train_dataset,
        exs,
        train_details,
        params,
    )

    sensealg = instantiate_sensealg(params.optimizer)
    optfun = Optimization.OptimizationFunction(
        (θ, P, fault_timespan_index) -> outer_loss_function(θ, fault_timespan_index),
        sensealg,
    )
    optprob = Optimization.OptimizationProblem(optfun, θ)

    cb = instantiate_cb!(
        output,
        params,
        validation_dataset,
        sys_validation,
        connecting_branches,
        surrogate,
        θ_ranges,
    )

    #Calculate loss before training
    loss, loss_initialization, loss_dynamic, surrogate_solution, fault_index =
        outer_loss_function(θ, (1, 1))

    @warn "Everything instantiated, starting solve with one epoch"
    timing_stats_compile = @timed Optimization.solve(
        optprob,
        optimizer,
        IterTools.ncycle(train_loader, 1),
        callback = cb,
    )
    push!(
        output["timing_stats_compile"],
        (
            time = timing_stats_compile.time,
            bytes = timing_stats_compile.bytes,
            gc_time = timing_stats_compile.gctime,
        ),
    )
    @warn "Starting full train with $per_solve_max_epochs epochs"
    timing_stats = @timed Optimization.solve(
        optprob,
        optimizer,
        IterTools.ncycle(train_loader, per_solve_max_epochs),
        callback = cb,
    )
    push!(
        output["timing_stats"],
        (
            time = timing_stats.time,
            bytes = timing_stats.bytes,
            gc_time = timing_stats.gctime,
        ),
    )
    res = timing_stats.value
    return res, output
end

function _generate_training_groups(fault_index, timespan_index, curriculum)
    x = [(f, t) for f in fault_index, t in timespan_index]
    dataset = reshape(x, length(x))
    if curriculum == "none"
        grouped_data = [dataset]
    elseif curriculum == "progressive"
        sorted = sort(dataset)
        grouped_data = [[x] for x in sorted]
    else
        @error "Curriculum not found"
        return false
    end

    return grouped_data
end

function _initialize_training_output_dict()
    return Dict{String, Any}(
        "loss" => DataFrames.DataFrame(
            Loss_initialization = Float64[],
            Loss_dynamic = Float64[],
            Loss = Float64[],
        ),
        "predictions" => DataFrames.DataFrame(
            parameters = Vector{Any}[],
            surrogate_solution = SteadyStateNeuralODE_solution[],
            fault_index = Tuple{Int64, Int64}[],
        ),
        "validation_loss" => DataFrames.DataFrame(
            mae_ir = Vector{Float64}[],
            max_error_ir = Vector{Float64}[],
            mae_ii = Vector{Float64}[],
            max_error_ii = Vector{Float64}[],
        ),
        "total_time" => [],
        "total_iterations" => 0,
        "recorded_iterations" => [],
        "final_loss" => Dict{String, Vector{Float64}}(),
        "timing_stats_compile" => [],
        "timing_stats" => [],
        "n_params_surrogate" => 0,
        "θ_ranges" => Dict{String, UnitRange{Int64}},
        "train_id" => "",
    )
end

function _capture_output(output_dict, output_directory, id)
    output_path = joinpath(output_directory, id)
    mkpath(output_path)
    for (key, value) in output_dict
        if typeof(value) == DataFrames.DataFrame
            df = pop!(output_dict, key)
            open(joinpath(output_path, key), "w") do io
                Arrow.write(io, df)
            end
            df = nothing
            GC.gc()
        end
    end
    open(joinpath(output_path, "high_level_outputs"), "w") do io
        JSON3.write(io, output_dict)
    end
end

function _calculate_per_solve_max_epochs(params, n_groups, n_samples)
    total_maxiters = params.maxiters
    per_solve_max_epochs = Int(floor(total_maxiters / n_groups / n_samples))
    if per_solve_max_epochs == 0
        @error "The calculated epochs per training group is 0. Adjust maxiters, the curriculum, or the size of the training dataset."
    end
    return per_solve_max_epochs
end
