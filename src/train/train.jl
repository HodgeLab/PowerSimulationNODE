
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
    exogenous_input_functions = []
    n_perturbations = length(train_data_params.perturbations)
    n_operating_points = length(train_data_params.operating_points)
    @assert n_perturbations * n_operating_points == length(train_dataset)
    for ix_o in 1:n_operating_points   #Note! Nested loop in same order as generate_surrogate_data from PSIDS
        for (ix_p, p) in enumerate(train_data_params.perturbations)
            ix = (ix_o - 1) * n_perturbations + ix_p
            train_data = train_dataset[ix]
            if train_data.stable == true    #only build exogenous inputs for the stable trajectories
                V = _surrogate_perturbation_to_function_of_time(
                    p,
                    train_data,
                    train_data_params,
                )
                push!(exogenous_input_functions, V)
            end
        end
    end
    return exogenous_input_functions
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
    if train_data_params.system == "full" || typeof(perturbation[1]) == PSIDS.Chirp
        for i in 1:size(data.surrogate_real_voltage)[1]
            function Vr(t)
                ix_after = findfirst(x -> x > t, data.tsteps)
                if ix_after === nothing  #correct behavior at end of timespan
                    ix_after = length(data.tsteps)
                end
                t_before = data.tsteps[ix_after - 1]
                t_after = data.tsteps[ix_after]
                val_before = data.surrogate_real_voltage[i, ix_after - 1]
                val_after = data.surrogate_real_voltage[i, ix_after]
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
                val_before = data.surrogate_imag_voltage[i, ix_after - 1]
                val_after = data.surrogate_imag_voltage[i, ix_after]
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

    return (t) -> [V_funcs[i](t) for i in eachindex(V_funcs)]
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
    Vr0_surrogate = data.surrogate_real_voltage[port_ix, 1]
    Vi0_surrogate = data.surrogate_imag_voltage[port_ix, 1]
    Vm0_surrogate = sqrt(Vr0_surrogate^2 + Vi0_surrogate^2)
    θ0_surrogate = atan(Vi0_surrogate / Vr0_surrogate)

    V_bias = Vm0_surrogate
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
    θ_bias = θ0_surrogate
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

#= function _single_perturbation_to_function_of_time(
    single_perturbation::PSIDS.Chirp,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
)
    @warn "The implemented Chirp voltage is technically the voltage behind the source impedance--- the assumption is that the source impedance is insignificantly small."
    Vr0 = data.surrogate_real_voltage[port_ix, 1]
    Vi0 = data.surrogate_imag_voltage[port_ix, 1]
    Vm0 = sqrt(Vr0^2 + Vi0^2)
    θ0 = atan(Vr0 / Vi0)

    tstart = single_perturbation.tstart
    N = single_perturbation.N
    ω1 = single_perturbation.ω1
    ω2 = single_perturbation.ω2
    V_amp = single_perturbation.V_amp
    ω_amp = single_perturbation.ω_amp

    function V(t)
        val = Vm0
        if t < tstart
            return val
        elseif t >= tstart && t < N
            val += V_amp * sin(ω1 * (t - tstart) + (ω2 - ω1) * (t - tstart)^2 / (2 * N))
            return val
        elseif t >= N #same expression as above, replace t with N
            val += V_amp * sin(ω1 * (N - tstart) + (ω2 - ω1) * (N - tstart)^2 / (2 * N))
            return val
        end
    end

    function θ(t)
        val = θ0
        if t < tstart
            return val
        elseif t >= tstart && t < N
            val +=
                t -
                ω_amp / ((ω1 + (ω2 - ω1) * (t - tstart) / N)) *
                cos(ω1 * (t - tstart) + (ω2 - ω1) * (t - tstart)^2 / (2 * N))
            return val
        elseif t >= N   #same expression as above, replace t with N
            val +=
                N -
                ω_amp / ((ω1 + (ω2 - ω1) * (N - tstart) / N)) *
                cos(ω1 * (N - tstart) + (ω2 - ω1) * (N - tstart)^2 / (2 * N))
            return val
        end
    end
    function Vr_func(t)
        return V(t) * cos(θ(t))
    end
    function Vi_func(t)
        return V(t) * sin(θ(t))
    end
    return (Vr_func, Vi_func)
end
 =#
function _single_perturbation_to_function_of_time(
    single_perturbation::PSIDS.VStep,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
)
    Vr0_surrogate = data.surrogate_real_voltage[port_ix, 1]
    Vi0_surrogate = data.surrogate_imag_voltage[port_ix, 1]
    Vm0_surrogate = sqrt(Vr0_surrogate^2 + Vi0_surrogate^2)
    θ0_surrogate = atan(Vi0_surrogate / Vr0_surrogate)
    function V(t)
        if t < single_perturbation.t_step
            return Vm0_surrogate
        else
            return single_perturbation.V_step
        end
    end
    function θ(t)
        return θ0_surrogate
    end

    function Vr_func(t)
        return V(t) * cos(θ(t))
    end
    function Vi_func(t)
        return V(t) * sin(θ(t))
    end
    return (Vr_func, Vi_func)
end

function visualize_loss(sys, θ, groundtruth_dataset, params, connecting_branches, θ_ranges)
    for s in PSY.get_components(PSIDS.SteadyStateNODEObs, sys)
        PSIDS.set_initializer_parameters!(s, θ[θ_ranges["initializer_range"]])
        PSIDS.set_node_parameters!(s, θ[θ_ranges["node_range"]])
        PSIDS.set_observer_parameters!(s, θ[θ_ranges["observation_range"]])
    end

    for s in PSY.get_components(PSIDS.SteadyStateNODE, sys)
        PSIDS.set_initializer_parameters!(s, θ[θ_ranges["initializer_range"]])
        PSIDS.set_node_parameters!(s, θ[θ_ranges["node_range"]])
    end
    operating_points = params.operating_points
    perturbations = params.perturbations
    generate_data_params = params.params
    surrogate_dataset = PSIDS.generate_surrogate_data(
        sys,
        sys,
        perturbations,
        operating_points,
        PSIDS.SteadyStateNODEDataParams(location_of_data_collection = connecting_branches),
        generate_data_params,
        dataset_aux = groundtruth_dataset,
    )
    @assert length(surrogate_dataset) == length(groundtruth_dataset)
    plots = []
    for ix in eachindex(surrogate_dataset, groundtruth_dataset)
        if groundtruth_dataset[ix].stable == true
            if surrogate_dataset[ix].stable == false
                @error "Groundtruth data is stable but surrogate is unstable for entry $ix of the dataset"
            elseif surrogate_dataset[ix].stable == true
                p1 = plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].surrogate_real_voltage,
                    label = "Vr (surr)",
                )
                plot!(
                    p1,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].surrogate_real_voltage,
                    label = "Vr (true)",
                )
                p2 = plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].surrogate_imag_voltage,
                    label = "Vi (surr)",
                )
                plot!(
                    p2,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].surrogate_imag_voltage,
                    label = "Vi (true)",
                )
                p3 = plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].real_current,
                    label = "Ir (surr)",
                )
                plot!(
                    p3,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].real_current,
                    label = "Ir (true)",
                )
                p4 = plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].imag_current,
                    label = "Ii (surr)",
                )
                plot!(
                    p4,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].imag_current,
                    label = "Ii (true)",
                )
                push!(plots, plot(p1, p2, p3, p4))
            end
        end
    end
    return plots
end

function evaluate_loss(sys, θ, groundtruth_dataset, params, connecting_branches, θ_ranges)
    for s in PSY.get_components(PSIDS.SteadyStateNODEObs, sys)
        PSIDS.set_initializer_parameters!(s, θ[θ_ranges["initializer_range"]])
        PSIDS.set_node_parameters!(s, θ[θ_ranges["node_range"]])
        PSIDS.set_observer_parameters!(s, θ[θ_ranges["observation_range"]])
    end

    for s in PSY.get_components(PSIDS.SteadyStateNODE, sys)
        PSIDS.set_initializer_parameters!(s, θ[θ_ranges["initializer_range"]])
        PSIDS.set_node_parameters!(s, θ[θ_ranges["node_range"]])
    end
    operating_points = params.operating_points
    perturbations = params.perturbations
    generate_data_params = params.params
    surrogate_dataset = PSIDS.generate_surrogate_data(
        sys,
        sys,
        perturbations,
        operating_points,
        PSIDS.SteadyStateNODEDataParams(location_of_data_collection = connecting_branches),
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
                push!(mae_ir, 0.0)   #Note: Cannot write Inf in Json spec, so assign 0 value if not stable (better for plotting too). Could lead to confusion if averaging over multiple plots. 
                push!(max_error_ir, 0.0)
                push!(mae_ii, 0.0)
                push!(max_error_ii, 0.0)
            elseif surrogate_dataset[ix].stable == true
                push!(
                    mae_ir,
                    mae(
                        surrogate_dataset[ix].real_current,
                        groundtruth_dataset[ix].real_current,
                    ),
                )
                push!(
                    max_error_ir,
                    maximum(
                        abs.(
                            surrogate_dataset[ix].real_current .-
                            groundtruth_dataset[ix].real_current,
                        ),
                    ),
                )
                push!(
                    mae_ii,
                    mae(
                        surrogate_dataset[ix].imag_current,
                        groundtruth_dataset[ix].imag_current,
                    ),
                )
                push!(
                    max_error_ii,
                    maximum(
                        abs.(
                            surrogate_dataset[ix].imag_current .-
                            groundtruth_dataset[ix].imag_current,
                        ),
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
    n_ports = size(train_dataset[1].real_current)[1]

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
            id_iq = PSID.ri_dq(θ0) * vcat(d.real_current, d.imag_current)
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
    params.train_time_limit_seconds += (floor(time()) - TIME_LIMIT_BUFFER_SECONDS)
    Random.seed!(params.rng_seed)

    #READ DATASETS 
    train_dataset = Serialization.deserialize(params.train_data_path)
    validation_dataset = Serialization.deserialize(params.validation_data_path)
    test_dataset = Serialization.deserialize(params.test_data_path)
    connecting_branches = Serialization.deserialize(params.data_collection_location_path)[2]
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

    @info "length(tstops) in first train condition: $(length(train_dataset[1].tstops))"
    @info "length(tsteps) in first train condition: $(length(train_dataset[1].tsteps))"
    n_ports = length(connecting_branches)
    @info "Surrogate contains $n_ports ports"
    output = _initialize_training_output_dict()
    θ = Float32[]
    try
        scaling_extrema = calculate_scaling_extrema(train_dataset)

        #READ VALIDATION SYSTEM AND ADD SURROGATE COMPONENT WITH STRUCTURE BASED ON PARAMS
        sys_validation = node_load_system(params.surrogate_system_path)
        sources = collect(
            PSY.get_components(
                PSY.Source,
                sys_validation,
                x -> PSY.get_name(x) !== "InfBus",
            ),
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
        PSY.to_json(sys_validation, params.surrogate_system_path, force = true) #Replace validation_system with a system that has the surrogate

        #INSTANTIATE 
        surrogate = instantiate_surrogate_flux(params, n_ports, scaling_extrema)

        p_nn_init, _ = Flux.destructure(surrogate)
        n_parameters = length(p_nn_init)
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
        exogenous_input_functions =
            _build_exogenous_input_functions(params.train_data, train_dataset)    #can build ex from the components in params.train_data or from the dataset values by interpolating...
        #want to test how this impacts the speed of a single train iteration (the interpolation)

        @assert length(train_dataset) == length(exogenous_input_functions)
        fault_indices = collect(1:length(train_dataset))    #TODO - should this be filtered for only stable faults? 
        n_samples = length(filter(x -> x.stable == true, train_dataset))
        @info "\n number of stable training samples: $n_samples"

        if isempty(params.p_start)
            p_full, _ = Flux.destructure(surrogate)
        else
            @assert length(p_train) == length(params.p_start)
            p_full = params.p_start
        end

        total_time = @elapsed begin
            for (opt_ix, opt) in enumerate(params.optimizer)
                p_fixed, p_train, p_map =
                    _initialize_params(opt.fix_params, p_full, surrogate)
                train_details = opt.curriculum_timespans
                timespan_indices = collect(1:length(opt.curriculum_timespans))
                train_groups = _generate_training_groups(
                    fault_indices,
                    timespan_indices,
                    opt.curriculum,
                )
                n_trains = length(train_groups)
                per_solve_max_epochs = _calculate_per_solve_max_epochs(
                    opt.maxiters,
                    n_trains,
                    length(train_groups[1]),
                )
                @warn opt
                algorithm = instantiate_optimizer(opt)
                sensealg = instantiate_sensealg(opt)
                @info "Curriculum pairings (fault_index, timespan_index)", train_groups
                @info "optimizer $(opt.algorithm)"
                @info "\n curriculum: $(opt.curriculum) \n # of solves: $n_trains \n # of epochs per training (based on maxiters parameter): $per_solve_max_epochs \n # of iterations per epoch $(length(train_groups[1])) \n # of samples per iteration $(length(train_groups[1][1])) "

                for group in train_groups
                    res, output = _train(
                        p_train,
                        p_fixed,
                        p_map,
                        surrogate,
                        θ_ranges,
                        connecting_branches,
                        train_dataset,
                        validation_dataset,
                        sys_validation,
                        exogenous_input_functions,
                        train_details,
                        params,
                        algorithm,
                        sensealg,
                        group,
                        per_solve_max_epochs,
                        output,
                        opt_ix,
                    )
                    p_train = res.u
                end
                p_full = vcat(p_fixed, p_train)[p_map]
            end
        end
        output["total_time"] = total_time

        output["final_loss"] = evaluate_loss(
            sys_validation,
            p_full,
            validation_dataset,
            params.validation_data,
            connecting_branches,
            θ_ranges,
        )

        _capture_output(output, params.output_data_path, params.train_id)
        return true, p_full
    catch e
        @error "Error in try block of train(): " exception = (e, catch_backtrace())
        _capture_output(output, params.output_data_path, params.train_id)
        return false, θ
    end
end

function _train(
    p_train::Vector{Float32},
    p_fixed::Vector{Float32},
    p_map::Vector{Int64},
    surrogate::SteadyStateNeuralODE,
    θ_ranges::Dict{String, UnitRange{Int64}},
    connecting_branches::Vector{Tuple{String, Symbol}},
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
    validation_dataset::Vector{PSIDS.SteadyStateNODEData},
    sys_validation::PSY.System,
    exogenous_input_functions,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
    algorithm::Union{
        OptimizationOptimisers.Optimisers.Adam,
        OptimizationOptimJL.Optim.AbstractOptimizer,
    },
    sensealg,
    group::Vector{Vector{Tuple{Int64, Int64}}},
    per_solve_max_epochs::Int,
    output::Dict{String, Any},
    opt_ix::Int64,
)
    @warn "Starting value of trainable parameters: $p_train"
    @warn "Starting value of fixed parameters: $p_fixed"
    train_loader =
        Flux.Data.DataLoader(group; batchsize = 1, shuffle = true, partial = true)
    outer_loss_function = instantiate_outer_loss_function(
        surrogate,
        train_dataset,
        exogenous_input_functions,
        train_details,
        p_fixed,
        p_map,
        params,
        opt_ix,
    )

    optfun = Optimization.OptimizationFunction(
        (p_train, P, vector_fault_timespan_index) ->
            outer_loss_function(p_train, vector_fault_timespan_index),
        sensealg,
    )
    optprob = Optimization.OptimizationProblem(optfun, p_train)

    cb = instantiate_cb!(
        output,
        params,
        validation_dataset,
        sys_validation,
        connecting_branches,
        surrogate,
        p_fixed,
        p_map,
        θ_ranges,
        opt_ix,
    )

    #Calculate loss before training - useful code for debugging changes to make sure forward pass works before checking train
    #=             loss, loss_initialization, loss_dynamic, surrogate_solution, fault_index_vector =
                    outer_loss_function(p_train, [(1, 1)])
                @warn loss 
                @warn loss_initialization
                @warn loss_dynamic 
                cb(p_train, loss, loss_initialization, loss_dynamic, surrogate_solution, fault_index_vector)   =#

    @warn "Starting full train: \n # of iterations per epoch: $(length(group)) \n # of epochs per solve: $per_solve_max_epochs \n max # of iterations for solve: $(per_solve_max_epochs*length(group))"
    timing_stats = @timed Optimization.solve(
        optprob,
        algorithm,
        IterTools.ncycle(train_loader, per_solve_max_epochs),
        callback = cb;
        allow_f_increases = true,
        show_trace = true,
        x_abstol = -1.0,
        x_reltol = -1.0,
        f_abstol = -1.0,
        f_reltol = -1.0,
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
    @warn "Residual of Optimization.solve(): $(res)"
    if res.original !== nothing
        @warn "The result from original optimization library: $(res.original)"
        @warn "stopped by?: $(res.original.stopped_by)"
    end
    return res, output
end

function _initialize_params(
    p_fixed::String,
    p_full::Vector{Float32},
    surrogate::SteadyStateNeuralODE,
)
    total_length = length(p_full)
    initializer_length = surrogate.len
    node_length = surrogate.len2
    observation_length = total_length - initializer_length - node_length
    if p_fixed == "initializer+observation"
        p_fixed = vcat(
            p_full[1:initializer_length],
            p_full[(initializer_length + node_length + 1):total_length],
        )
        p_train = p_full[(initializer_length + 1):(initializer_length + node_length)]
        p_map = vcat(
            1:initializer_length,
            (total_length - node_length + 1):total_length,
            (initializer_length + 1):(initializer_length + observation_length),
        )  #WRONG?
        return p_fixed, p_train, p_map
    elseif p_fixed == "initializer"
        p_fixed = p_full[1:initializer_length]
        p_train = p_full[(initializer_length + 1):end]
        p_map = collect(1:total_length)
        return p_fixed, p_train, p_map
    elseif p_fixed == "none"
        p_fixed = Float32[]
        p_train = p_full
        p_map = collect(1:length(p_full))
        return p_fixed, p_train, p_map
    else
        @error "invalid entry for parameter p_fixed which indicates which types of parameters should be held constant"
        return false
    end
end

function _generate_training_groups(fault_index, timespan_index, curriculum)
    if curriculum == "individual faults"
        x = [[(f, t)] for f in fault_index, t in timespan_index]
        dataset = reshape(x, length(x))
        grouped_data = [dataset]
        return grouped_data
    elseif curriculum == "individual faults x2"
        x = [[(f, t)] for f in fault_index, t in timespan_index]
        dataset = reshape(x, length(x))
        grouped_data = [dataset, dataset]
        return grouped_data
    elseif curriculum == "individual faults x3"
        x = [[(f, t)] for f in fault_index, t in timespan_index]
        dataset = reshape(x, length(x))
        grouped_data = [dataset, dataset, dataset]
        return grouped_data
    elseif curriculum == "simultaneous"
        x = [(f, t) for f in fault_index, t in timespan_index]
        dataset = reshape(x, length(x))
        grouped_data = [[dataset]]
        return grouped_data
        #=     elseif curriculum == "progressive"
                x = [[(f, t)] for f in fault_index, t in timespan_index]
                dataset = reshape(x, length(x))
                sorted = sort(dataset)
                grouped_data = [[x] for x in sorted]
                return grouped_data  =#
    else
        @error "Curriculum not found"
        return false
    end
end

function _initialize_training_output_dict()
    return Dict{String, Any}(
        "loss" => DataFrames.DataFrame(
            Loss_initialization = Float64[],
            Loss_dynamic = Float64[],
            Loss = Float64[],
            reached_ss = Bool[],
        ),
        "predictions" => DataFrames.DataFrame(
            parameters = Vector{Any}[],
            surrogate_solution = SteadyStateNeuralODE_solution[],
            fault_index = Vector{Tuple{Int64, Int64}}[],
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

function _calculate_per_solve_max_epochs(total_maxiters, n_groups, n_samples)
    per_solve_max_epochs = Int(floor(total_maxiters / n_groups / n_samples))
    if per_solve_max_epochs == 0
        @error "The calculated epochs per training group is 0. Adjust maxiters, the curriculum, or the size of the training dataset."
    end
    return per_solve_max_epochs
end
