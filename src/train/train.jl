
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
            if train_data.stable == true
                V = _surrogate_perturbation_to_function_of_time(
                    p,
                    train_data,
                    train_data_params,
                )
                push!(exogenous_input_functions, V)
            else
                push!(exogenous_input_functions, x -> x)  #add dummy function for unstable trajectories so same index that is used for dataset can be used for exogenous_input_functions.
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
    #NOTE: Cannot write a closed form expression for V(t) for the Chirp source so instead just interpolate between data points like we do for real faults.
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

"""
    function generate_surrogate_dataset(
        sys_main,
        sys_aux,
        θ,
        groundtruth_dataset,
        data_params,
        data_collection_location,
        model_params,
    )

# Fields
- `sys_main`: System with surrogate already included (usually the modified validation system)
- `sys_aux`: Auxiliary system for finding components to use for perturbation (can be unmodified validation system)
- `θ`: New surrogate parameters to evaluate loss for
- `groundtruth_dataset`: ground truth data for evaluating loss.
- `data_params`: the parameters used to generate `groundtruth_dataset`
- `data_collection_location::Vector{Tuple{String, Symbol}}`: the data collection location for generating data. A vector of Tuples of branch name and either `:to` or `:from` for interpreting the polarity of data from those branches.
- `model_params`: model params of the surrogate. 
"""
function generate_surrogate_dataset(
    sys_main,
    sys_aux,
    θ,
    groundtruth_dataset,
    data_params,
    data_collection_location,
    model_params,
)
    a = time()
    parameterize_surrogate_psid!(sys_main, θ, model_params)

    operating_points = data_params.operating_points
    perturbations = data_params.perturbations
    generate_data_params = data_params.params
    surrogate_dataset = PSIDS.generate_surrogate_data(
        sys_main,
        sys_aux,
        perturbations,
        operating_points,
        PSIDS.SteadyStateNODEDataParams(
            location_of_data_collection = data_collection_location,
        ),
        generate_data_params,
        dataset_aux = groundtruth_dataset,
        surrogate_params = model_params,
    )
    @warn "generate_surrogate_dataset time (s):  ", time() - a
    return surrogate_dataset
end

"""
    function evaluate_loss(
        surrogate_dataset, 
        groundtruth_dataset,
    ) 
"""
function evaluate_loss(surrogate_dataset, groundtruth_dataset)
    @assert length(surrogate_dataset) == length(groundtruth_dataset)
    mae_ir = Float64[]
    max_error_ir = Float64[]
    mae_ii = Float64[]
    max_error_ii = Float64[]
    for ix in eachindex(surrogate_dataset, groundtruth_dataset)
        if groundtruth_dataset[ix].stable == true
            if surrogate_dataset[ix].stable == false
                push!(mae_ir, 0.0)   #Note:  Cannot write Inf in Json spec, so assign 0 value if not stable (better for plotting too). Could lead to confusion if averaging over multiple plots. 
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
    function visualize_loss(
        surrogate_dataset, 
        groundtruth_dataset,
    )
"""
function visualize_loss(surrogate_dataset, groundtruth_dataset)
    @assert length(surrogate_dataset) == length(groundtruth_dataset)
    plots = []
    for ix in eachindex(surrogate_dataset, groundtruth_dataset)
        if groundtruth_dataset[ix].stable == true
            if surrogate_dataset[ix].stable == false
                @error "Groundtruth data is stable but surrogate is unstable for entry $ix of the dataset"
            elseif surrogate_dataset[ix].stable == true
                p1 = Plots.plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].surrogate_real_voltage',
                    label = "Vr (surr)",
                )
                Plots.plot!(
                    p1,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].surrogate_real_voltage',
                    label = "Vr (true)",
                )
                p2 = Plots.plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].surrogate_imag_voltage',
                    label = "Vi (surr)",
                )
                Plots.plot!(
                    p2,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].surrogate_imag_voltage',
                    label = "Vi (true)",
                )
                p3 = Plots.plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].real_current',
                    label = "Ir (surr)",
                )
                Plots.plot!(
                    p3,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].real_current',
                    label = "Ir (true)",
                )
                p4 = Plots.plot(
                    surrogate_dataset[ix].tsteps,
                    surrogate_dataset[ix].imag_current',
                    label = "Ii (surr)",
                )
                Plots.plot!(
                    p4,
                    groundtruth_dataset[ix].tsteps,
                    groundtruth_dataset[ix].imag_current',
                    label = "Ii (true)",
                )
                push!(plots, Plots.plot(p1, p2, p3, p4))
            end
        end
    end
    return plots
end

function _calculate_n_params(structure)
    n_params = 0
    for layer in structure
        if layer[3] == true #bias layer 
            n_params += (layer[1] + 1) * layer[2]
        else
            n_params += layer[1] * layer[2]
        end
    end
    return n_params
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float32},
    model_params::PSIDS.SteadyStateNODEParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    surrogate = PSY.get_component(PSIDS.SteadyStateNODE, sys, model_params.name)
    n_params_initializer = _calculate_n_params(surrogate.initializer_structure)
    n_params_node = _calculate_n_params(surrogate.node_structure)
    @assert length(θ) == n_params_initializer + n_params_node
    PSIDS.set_initializer_parameters!(surrogate, θ[1:n_params_initializer])
    PSIDS.set_node_parameters!(
        surrogate,
        θ[(n_params_initializer + 1):(n_params_initializer + n_params_node)],
    )
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float32},
    model_params::PSIDS.SteadyStateNODEObsParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    surrogate = PSY.get_component(PSIDS.SteadyStateNODEObs, sys, model_params.name)
    n_params_initializer = _calculate_n_params(surrogate.initializer_structure)
    n_params_node = _calculate_n_params(surrogate.node_structure)
    n_params_observation = _calculate_n_params(surrogate.observer_structure)
    @assert length(θ) == n_params_initializer + n_params_node + n_params_observation
    PSIDS.set_initializer_parameters!(surrogate, θ[1:n_params_initializer])
    PSIDS.set_node_parameters!(
        surrogate,
        θ[(n_params_initializer + 1):(n_params_initializer + n_params_node)],
    )
    PSIDS.set_observer_parameters!(
        surrogate,
        θ[(n_params_initializer + n_params_node + 1):(n_params_initializer + n_params_node + n_params_observation)],
    )
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.ClassicGenParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    static = PSY.get_component(PSY.StaticInjection, sys, model_params.name)
    PSY.set_active_power_limits!(static, (min = -max_P, max = max_P))
    PSY.set_reactive_power_limits!(static, (min = -max_Q, max = max_Q))

    R, Xd_p, eq_p = θ[1:3]  #machine parameters 
    H, D = θ[4:5]   #shaft parameters 

    surrogate = PSY.get_component(PSY.DynamicInjection, sys, model_params.name)
    machine = PSY.get_machine(surrogate)
    PSY.set_R!(machine, R)
    PSY.set_Xd_p!(machine, Xd_p)
    PSY.set_eq_p!(machine, eq_p)

    shaft = PSY.get_shaft(surrogate)
    PSY.set_H!(shaft, H)
    PSY.set_D!(shaft, D)
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.GFLParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    static = PSY.get_component(PSY.StaticInjection, sys, model_params.name)
    PSY.set_active_power_limits!(static, (min = -max_P, max = max_P))
    PSY.set_reactive_power_limits!(static, (min = -max_Q, max = max_Q))

    #NOTE: references are set during initialization 
    surrogate = PSY.get_component(PSY.DynamicInjection, sys, model_params.name)
    converter = PSY.get_converter(surrogate)
    PSY.set_rated_voltage!(converter, θ[gfl_indices[:params][:rated_voltage_gfl]])
    PSY.set_rated_current!(converter, θ[gfl_indices[:params][:rated_current_gfl]])

    active_power = PSY.get_active_power_control(PSY.get_outer_control(surrogate))
    PSY.set_Kp_p!(active_power, θ[gfl_indices[:params][:Kp_p]])
    PSY.set_Ki_p!(active_power, θ[gfl_indices[:params][:Ki_p]])
    PSY.set_ωz!(active_power, θ[gfl_indices[:params][:ωz_gfl]])
    #PSY.set_P_ref!(active_power, P_ref)

    reactive_power = PSY.get_reactive_power_control(PSY.get_outer_control(surrogate))
    PSY.set_Kp_q!(reactive_power, θ[gfl_indices[:params][:Kp_q]])
    PSY.set_Ki_q!(reactive_power, θ[gfl_indices[:params][:Ki_q]])
    PSY.set_ωf!(reactive_power, θ[gfl_indices[:params][:ωf_gfl]])
    #PSY.set_V_ref!(reactive_power, V_ref)
    #PSY.set_Q_ref!(reactive_power, Q_ref)

    inner_control = PSY.get_inner_control(surrogate)
    PSY.set_kpc!(inner_control, θ[gfl_indices[:params][:kpc_gfl]])
    PSY.set_kic!(inner_control, θ[gfl_indices[:params][:kic_gfl]])
    PSY.set_kffv!(inner_control, θ[gfl_indices[:params][:kffv_gfl]])

    dc_source = PSY.get_dc_source(surrogate)
    PSY.set_voltage!(dc_source, θ[gfl_indices[:params][:voltage_gfl]])

    freq_estimator = PSY.get_freq_estimator(surrogate)
    PSY.set_ω_lp!(freq_estimator, θ[gfl_indices[:params][:ω_lp]])
    PSY.set_kp_pll!(freq_estimator, θ[gfl_indices[:params][:kp_pll]])
    PSY.set_ki_pll!(freq_estimator, θ[gfl_indices[:params][:ki_pll]])

    filter = PSY.get_filter(surrogate)
    PSY.set_lf!(filter, θ[gfl_indices[:params][:lf_gfl]])
    PSY.set_rf!(filter, θ[gfl_indices[:params][:rf_gfl]])
    PSY.set_cf!(filter, θ[gfl_indices[:params][:cf_gfl]])
    PSY.set_lg!(filter, θ[gfl_indices[:params][:lg_gfl]])
    PSY.set_rg!(filter, θ[gfl_indices[:params][:rg_gfl]])
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.GFMParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    static = PSY.get_component(PSY.StaticInjection, sys, model_params.name)
    PSY.set_active_power_limits!(static, (min = -max_P, max = max_P))
    PSY.set_reactive_power_limits!(static, (min = -max_Q, max = max_Q))

    #NOTE: references are set during initialization 
    surrogate = PSY.get_component(PSY.DynamicInjection, sys, model_params.name)
    converter = PSY.get_converter(surrogate)
    PSY.set_rated_voltage!(converter, θ[gfm_indices[:params][:rated_voltage_gfm]])
    PSY.set_rated_current!(converter, θ[gfm_indices[:params][:rated_current_gfm]])

    active_power = PSY.get_active_power_control(PSY.get_outer_control(surrogate))
    PSY.set_Rp!(active_power, θ[gfm_indices[:params][:Rp]])
    PSY.set_ωz!(active_power, θ[gfm_indices[:params][:ωz_gfm]])
    #PSY.set_P_ref!(active_power, P_ref)

    reactive_power = PSY.get_reactive_power_control(PSY.get_outer_control(surrogate))
    PSY.set_kq!(reactive_power, θ[gfm_indices[:params][:kq]])
    PSY.set_ωf!(reactive_power, θ[gfm_indices[:params][:ωf_gfm]])
    #PSY.set_V_ref!(reactive_power, V_ref)

    inner_control = PSY.get_inner_control(surrogate)
    PSY.set_kpv!(inner_control, θ[gfm_indices[:params][:kpv]])
    PSY.set_kiv!(inner_control, θ[gfm_indices[:params][:kiv]])
    PSY.set_kffv!(inner_control, θ[gfm_indices[:params][:kffv_gfm]])
    PSY.set_rv!(inner_control, θ[gfm_indices[:params][:rv]])
    PSY.set_lv!(inner_control, θ[gfm_indices[:params][:lv]])
    PSY.set_kpc!(inner_control, θ[gfm_indices[:params][:kpc_gfm]])
    PSY.set_kic!(inner_control, θ[gfm_indices[:params][:kic_gfm]])
    PSY.set_kffi!(inner_control, θ[gfm_indices[:params][:kffi]])
    PSY.set_ωad!(inner_control, θ[gfm_indices[:params][:ωad]])
    PSY.set_kad!(inner_control, θ[gfm_indices[:params][:kad]])

    dc_source = PSY.get_dc_source(surrogate)
    PSY.set_voltage!(dc_source, θ[gfm_indices[:params][:voltage_gfm]])

    freq_estimator = PSY.get_freq_estimator(surrogate)

    filter = PSY.get_filter(surrogate)
    PSY.set_lf!(filter, θ[gfm_indices[:params][:lf_gfm]])
    PSY.set_rf!(filter, θ[gfm_indices[:params][:rf_gfm]])
    PSY.set_cf!(filter, θ[gfm_indices[:params][:cf_gfm]])
    PSY.set_lg!(filter, θ[gfm_indices[:params][:lg_gfm]])
    PSY.set_rg!(filter, θ[gfm_indices[:params][:rg_gfm]])
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.ZIPParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    #Distributed max_P and max_Q according to the zip specific parameters. 
    P_total =
        θ[zip_indices[:params][:max_active_power_Z]] +
        θ[zip_indices[:params][:max_active_power_I]] +
        θ[zip_indices[:params][:max_active_power_P]]
    Q_total =
        θ[zip_indices[:params][:max_reactive_power_Z]] +
        θ[zip_indices[:params][:max_reactive_power_I]] +
        θ[zip_indices[:params][:max_reactive_power_P]]
    load = PSY.get_component(PSY.StandardLoad, sys, model_params.name) # string(model_params.name, "_Z"))
    PSY.set_max_impedance_active_power!(
        load,
        max_P * θ[zip_indices[:params][:max_active_power_Z]] / P_total,
    )
    PSY.set_max_impedance_reactive_power!(
        load,
        max_Q * θ[zip_indices[:params][:max_reactive_power_Z]] / Q_total,
    )
    PSY.set_max_current_active_power!(
        load,
        max_P * θ[zip_indices[:params][:max_active_power_I]] / P_total,
    )
    PSY.set_max_current_reactive_power!(
        load,
        max_Q * θ[zip_indices[:params][:max_reactive_power_I]] / Q_total,
    )
    PSY.set_max_constant_active_power!(
        load,
        max_P * θ[zip_indices[:params][:max_active_power_P]] / P_total,
    )
    PSY.set_max_constant_reactive_power!(
        load,
        max_Q * θ[zip_indices[:params][:max_reactive_power_P]] / Q_total,
    )
end

function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.MultiDeviceParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    n_maxpowers_params =
        2 * (length(model_params.static_devices) + length(model_params.dynamic_devices))
    θ_maxpowers = θ[1:n_maxpowers_params]
    θ_devices = θ[(n_maxpowers_params + 1):end]

    ix_maxpowers = 1
    ix_devices_start = 1
    for s in model_params.static_devices
        ix_devices_end = ix_devices_start + n_params(s) - 1
        parameterize_surrogate_psid!(
            sys,
            θ_devices[ix_devices_start:ix_devices_end],
            s;
            max_P = θ_maxpowers[ix_maxpowers],
            max_Q = θ_maxpowers[ix_maxpowers + 1],
        )
        ix_maxpowers += 2
        ix_devices_start = ix_devices_end + 1
    end
    for s in model_params.dynamic_devices
        ix_devices_end = ix_devices_start + n_params(s) - 1
        parameterize_surrogate_psid!(
            sys,
            θ_devices[ix_devices_start:ix_devices_end],
            s;
            max_P = θ_maxpowers[ix_maxpowers],
            max_Q = θ_maxpowers[ix_maxpowers + 1],
        )
        ix_maxpowers += 2
        ix_devices_start = ix_devices_end + 1
    end
end

#= function parameterize_surrogate_psid!(
    sys::PSY.System,
    θ::Vector{Float64},
    model_params::PSIDS.MultiDeviceLineParams;
    max_P = 1.0,
    max_Q = 1.0,
)
    n_maxpowers_params =
        2 * (length(model_params.static_devices) + length(model_params.dynamic_devices))
    θ_maxpowers = θ[4:(3 + n_maxpowers_params)]
    θ_line = θ[1:3] #θ[(n_maxpowers_params + 1):(n_maxpowers_params + 3)]
    θ_devices = θ[(n_maxpowers_params + 4):end]

    line = PSY.get_component(PSY.Component, sys, string(model_params.name, "-line"))
    PSY.set_r!(line, θ_line[1])
    PSY.set_x!(line, θ_line[2])
    PSY.set_b!(line, (from = θ_line[3], to = θ_line[3]))

    ix_maxpowers = 1
    ix_devices_start = 1
    for s in model_params.static_devices
        ix_devices_end = ix_devices_start + n_params(s) - 1
        parameterize_surrogate_psid!(
            sys,
            θ_devices[ix_devices_start:ix_devices_end],
            s;
            max_P = θ_maxpowers[ix_maxpowers],
            max_Q = θ_maxpowers[ix_maxpowers + 1],
        )
        ix_maxpowers += 2
        ix_devices_start = ix_devices_end + 1
    end
    for s in model_params.dynamic_devices
        ix_devices_end = ix_devices_start + n_params(s) - 1
        parameterize_surrogate_psid!(
            sys,
            θ_devices[ix_devices_start:ix_devices_end],
            s;
            max_P = θ_maxpowers[ix_maxpowers],
            max_Q = θ_maxpowers[ix_maxpowers + 1],
        )
        ix_maxpowers += 2
        ix_devices_start = ix_devices_end + 1
    end
end =#

function _check_dimensionality(
    data_collection_location,
    model_params::PSIDS.ClassicGenParams,
)
    @assert length(data_collection_location) == 1
end

function _check_dimensionality(data_collection_location, model_params::PSIDS.GFLParams)
    @assert length(data_collection_location) == 1
end

function _check_dimensionality(data_collection_location, model_params::PSIDS.GFMParams)
    @assert length(data_collection_location) == 1
end

function _check_dimensionality(data_collection_location, model_params::PSIDS.ZIPParams)
    @assert length(data_collection_location) == 1
end

function _check_dimensionality(
    data_collection_location,
    model_params::PSIDS.MultiDeviceParams,
)
    @assert length(data_collection_location) == 1
end

function _check_dimensionality(
    data_collection_location,
    model_params::PSIDS.SteadyStateNODEParams,
)
    @assert model_params.n_ports == length(data_collection_location)
end

function _check_dimensionality(
    data_collection_location,
    model_params::PSIDS.SteadyStateNODEObsParams,
)
    @assert model_params.n_ports == length(data_collection_location)
end

function _check_surrogate_convergence(
    surrogate,
    train_dataset,
    steadystate_solver,
    dynamic_solver,
    args...;
    kwargs...,
)
    return true
end

#Check all entries in dataset converge in under 10 iterations of DEQ layer. 
function _check_surrogate_convergence(
    surrogate::SteadyStateNeuralODELayer,
    train_dataset,
    steadystate_solver,
    dynamic_solver,
    args...;
    kwargs...,
)
    train_dataset_stable = filter(x -> x.stable, train_dataset)
    v0s = [
        [entry.surrogate_real_voltage[1], entry.surrogate_imag_voltage[1]] for
        entry in train_dataset_stable
    ]
    i0s = [[entry.real_current[1], entry.imag_current[1]] for entry in train_dataset_stable]
    converged = []
    iterations = []
    for (i, v0) in enumerate(v0s)
        i0 = i0s[i]
        sol = surrogate(
            (t) -> v0,
            v0,
            i0,
            [0.0, 1.0],
            [0.0, 1.0],
            steadystate_solver,
            dynamic_solver,
            args...;
            kwargs...,
        )
        push!(converged, sol.converged)
        push!(iterations, sol.deq_iterations)
    end
    @warn "checking convergence of initial surrogate: convergence for each train dataset entry : $converged, iterations for each train dataset entry: $iterations"
    if sum(converged) / length(converged) == 1.0 && !(any(x -> x > 10, iterations))
        return true
    else
        return false
    end
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
    data_collection_location_validation =
        Serialization.deserialize(params.data_collection_location_path)[2]
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

    #Commented because fails for multiple parallel lines connecting surrogate to system 
    #_check_dimensionality(data_collection_location_validation, params.model_params)

    output = _initialize_training_output_dict(params.model_params)
    try
        sys_validation = node_load_system(params.surrogate_system_path)
        sys_validation_aux = deepcopy(sys_validation)
        add_surrogate_psid!(sys_validation, params.model_params, train_dataset)
        PSY.to_json(sys_validation, params.modified_surrogate_system_path, force = true)

        #INSTANTIATE 
        surrogate = instantiate_surrogate_flux(params, params.model_params, train_dataset)
        dynamic_reltol = params.optimizer[1].dynamic_solver.reltol
        dynamic_abstol = params.optimizer[1].dynamic_solver.abstol
        dynamic_maxiters = params.optimizer[1].dynamic_solver.maxiters
        steadystate_abstol = params.optimizer[1].steadystate_solver.abstol
        steadystate_solver = PowerSimulationNODE.instantiate_steadystate_solver(
            params.optimizer[1].steadystate_solver,
        )
        dynamic_solver =
            PowerSimulationNODE.instantiate_solver(params.optimizer[1].dynamic_solver)

        for i in 1:ATTEMPTS_TO_FIND_CONVERGENT_SURROGATE
            if _check_surrogate_convergence(
                surrogate,
                train_dataset,
                steadystate_solver,
                dynamic_solver,
                steadystate_abstol;
                reltol = dynamic_reltol,
                abstol = dynamic_abstol,
                maxiters = dynamic_maxiters,
            ) == true
                break
            elseif i == ATTEMPTS_TO_FIND_CONVERGENT_SURROGATE
                @error "DEQ did not converge for $i different initializations"
                break
            end
            surrogate =
                instantiate_surrogate_flux(params, params.model_params, train_dataset)
        end

        p_nn_init, _ = Flux.destructure(surrogate)
        n_parameters = length(p_nn_init)
        @info "Surrogate has $n_parameters parameters"
        res = nothing
        output["train_id"] = params.train_id
        output["n_params_surrogate"] = n_parameters
        exogenous_input_functions =
            _build_exogenous_input_functions(params.train_data, train_dataset)    #can build ex from the components in params.train_data or from the dataset values by interpolating

        @assert length(train_dataset) == length(exogenous_input_functions)
        stable_fault_indices =
            indexin(filter(x -> x.stable == true, train_dataset), train_dataset)    #Only train on stable faults from train_dataset
        n_samples = length(filter(x -> x.stable == true, train_dataset))
        @info "\n number of stable training samples: $n_samples"

        if isempty(params.p_start)
            p_full, _ = Flux.destructure(surrogate)
        else
            p_full = params.p_start
        end
        total_time = @elapsed begin
            for (opt_ix, opt) in enumerate(params.optimizer)
                p_fixed, p_train, p_map =
                    _initialize_params(opt.fix_params, p_full, surrogate)
                train_details = opt.curriculum_timespans
                timespan_indices = collect(1:length(opt.curriculum_timespans))
                train_groups = _generate_training_groups(
                    stable_fault_indices,
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

                dynamic_reltol = opt.dynamic_solver.reltol
                dynamic_abstol = opt.dynamic_solver.abstol
                dynamic_maxiters = opt.dynamic_solver.maxiters
                steadystate_abstol = opt.steadystate_solver.abstol
                steadystate_solver = PowerSimulationNODE.instantiate_steadystate_solver(
                    opt.steadystate_solver,
                )
                dynamic_solver =
                    PowerSimulationNODE.instantiate_solver(opt.dynamic_solver)

                for group in train_groups
                    p_train, output = _train(
                        p_train,
                        p_fixed,
                        p_map,
                        surrogate,
                        data_collection_location_validation,
                        train_dataset,
                        validation_dataset,
                        sys_validation,
                        sys_validation_aux,
                        exogenous_input_functions,
                        train_details,
                        params,
                        algorithm,
                        sensealg,
                        group,
                        per_solve_max_epochs,
                        output,
                        opt_ix,
                        steadystate_solver,
                        dynamic_solver,
                        steadystate_abstol;
                        reltol = dynamic_reltol,
                        abstol = dynamic_abstol,
                        maxiters = dynamic_maxiters,
                    )
                end
                p_full = vcat(p_fixed, p_train)[p_map]
            end
        end
        output["total_time"] = total_time

        surrogate_dataset = generate_surrogate_dataset(
            sys_validation,
            sys_validation_aux,
            p_full,
            validation_dataset,
            params.validation_data,
            data_collection_location_validation,
            params.model_params,
        )

        output["final_loss"] = evaluate_loss(surrogate_dataset, validation_dataset)
        _capture_output(output, params.output_data_path, params.train_id)
        return true, p_full
    catch e
        @error "Error in try block of train(): " exception = (e, catch_backtrace())
        _capture_output(output, params.output_data_path, params.train_id)
        return false, []
    end
end

function _train(
    p_train::Union{Vector{Float32}, Vector{Float64}},
    p_fixed::Union{Vector{Float32}, Vector{Float64}},
    p_map::Vector{Int64},
    surrogate::Union{SteadyStateNeuralODE, ClassicGen, GFL, GFM, ZIP, MultiDevice},
    data_collection_location::Vector{Tuple{String, Symbol}},
    train_dataset::Vector{PSIDS.SteadyStateNODEData},
    validation_dataset::Vector{PSIDS.SteadyStateNODEData},
    sys_validation::PSY.System,
    sys_validation_aux::PSY.System,
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
    ss_solver,  #add types 
    dyn_solver,
    args...;
    kwargs...,
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
        ss_solver,
        dyn_solver,
        args...;
        kwargs...,
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
        sys_validation_aux,
        data_collection_location,
        surrogate,
        p_fixed,
        p_map,
        opt_ix,
    )

    #Calculate loss before training - useful code for debugging changes to make sure forward pass works before checking train (don't run callback because it records data)
    loss, loss_initialization, loss_dynamic, surrogate_solution, fault_index_vector =
        outer_loss_function(p_train, [(1, 1)])

    @warn "forward pass initialization loss:   ", loss_initialization
    @warn "forward pass dynamic loss:   ", loss_dynamic
    @warn "forward pass loss:   ", loss

    @warn "Starting full train: \n # of iterations per epoch: $(length(group)) \n # of epochs per solve: $per_solve_max_epochs \n max # of iterations for solve: $(per_solve_max_epochs*length(group))"
    if typeof(algorithm) <: Optim.AbstractOptimizer
        timing_stats = @timed Optimization.solve(
            optprob,
            algorithm,
            IterTools.ncycle(train_loader, per_solve_max_epochs),
            callback = cb;
            allow_f_increases = true,
            show_trace = false,
            x_abstol = -1.0,
            x_reltol = -1.0,
            f_abstol = -1.0,
            f_reltol = -1.0,
        )
    else
        timing_stats = @timed Optimization.solve(
            optprob,
            algorithm,
            IterTools.ncycle(train_loader, per_solve_max_epochs),
            callback = cb;
            save_best = false,
            allow_f_increases = true,
            show_trace = false,
            x_abstol = -1.0,
            x_reltol = -1.0,
            f_abstol = -1.0,
            f_reltol = -1.0,
        )
    end
    push!(
        output["timing_stats"],
        (
            time = timing_stats.time,
            bytes = timing_stats.bytes,
            gc_time = timing_stats.gctime,
        ),
    )
    @warn "chosen iteration: ", output["chosen_iteration"]
    @warn "total iterations: ", output["total_iterations"]

    chosen_iteration_index =
        indexin(output["chosen_iteration"], output["recorded_iterations"])[1]
    chosen_trainable_parameters =
        output["predictions"][chosen_iteration_index, "trainable_parameters"][1]
    res = timing_stats.value
    #@assert res_alt[1] == res.u does not pass because the Optimization.solve returns the parameters that are updated once more after the ones passed to the callback. Not sure which is correct, shouldn't matter much.
    if res.original !== nothing
        @warn "The result from original optimization library: $(res.original)"
        @warn "stopped by?: $(res.original.stopped_by)"
    end
    return chosen_trainable_parameters, output
end

function _initialize_params(
    p_fixed::Vector{Symbol},
    p_full,
    surrogate::SteadyStateNeuralODE,
)
    @warn typeof(p_full)    #p_full comes as a typeo f any 
    total_length = length(p_full)
    initializer_length = surrogate.len
    node_length = surrogate.len2
    observation_length = total_length - initializer_length - node_length
    if (:initializer in p_fixed) && (:observation in p_fixed)
        p_fix = vcat(
            p_full[1:initializer_length],
            p_full[(initializer_length + node_length + 1):total_length],
        )
        p_train = p_full[(initializer_length + 1):(initializer_length + node_length)]
        p_map = vcat(
            1:initializer_length,
            (total_length - node_length + 1):total_length,
            (initializer_length + 1):(initializer_length + observation_length),
        )  #WRONG?
        return p_fix, p_train, p_map
    elseif (:initializer in p_fixed)
        p_fix = Float32.(p_full[1:initializer_length])
        p_train = Float32.(p_full[(initializer_length + 1):end])
        p_map = collect(1:total_length)
        return p_fix, p_train, p_map
    elseif p_fixed == []
        p_fix = Float32[]
        p_train = Float32.(p_full)
        p_map = collect(1:length(p_full))
        return p_fix, p_train, p_map
    else
        @error "invalid entry for parameter p_fixed which indicates which types of parameters should be held constant"
        return false
    end
end

function _initialize_params(p_fixed::Vector{Symbol}, p_full, surrogate::ClassicGen)
    @assert !(nothing in indexin(p_fixed, [:R, :Xd_p, :eq_p, :H, :D]))  #ensure given p_fixed is a valid parameter
    @info "original parameter vector: $p_full"
    p_fix = Float64[]
    p_train = Float64[]
    fixed_indices = []
    train_indices = []
    for i in 1:length(p_full)
        if i in indexin(p_fixed, [:R, :Xd_p, :eq_p, :H, :D])
            push!(p_fix, p_full[i])
            push!(fixed_indices, i)
        else
            push!(p_train, p_full[i])
            push!(train_indices, i)
        end
    end
    p_map = Vector{Int64}(undef, 5)
    for (ix, i) in enumerate(fixed_indices)
        p_map[i] = ix
    end
    for (ix, i) in enumerate(train_indices)
        p_map[i] = ix + length(fixed_indices)
    end
    @info "remapped parameter vector: $(vcat(p_fix, p_train)[p_map])"
    return p_fix, p_train, p_map
end

function _initialize_params(p_fixed::Vector{Symbol}, p_full, surrogate::GFL)
    param_symbols = ordered_param_symbols(surrogate)
    @assert !(nothing in indexin(p_fixed, param_symbols))  #ensure given p_fixed is a valid parameter
    @info "original parameter vector: $p_full"
    p_fix = Float64[]
    p_train = Float64[]
    fixed_indices = []
    train_indices = []
    for i in 1:length(p_full)
        if i in indexin(p_fixed, param_symbols)
            push!(p_fix, p_full[i])
            push!(fixed_indices, i)
        else
            push!(p_train, p_full[i])
            push!(train_indices, i)
        end
    end
    p_map = Vector{Int64}(undef, length(param_symbols))
    for (ix, i) in enumerate(fixed_indices)
        p_map[i] = ix
    end
    for (ix, i) in enumerate(train_indices)
        p_map[i] = ix + length(fixed_indices)
    end
    @info "remapped parameter vector: $(vcat(p_fix, p_train)[p_map])"
    return p_fix, p_train, p_map
end

function _initialize_params(p_fixed::Vector{Symbol}, p_full, surrogate::GFM)
    param_symbols = ordered_param_symbols(surrogate)
    @assert !(nothing in indexin(p_fixed, param_symbols))  #ensure given p_fixed is a valid parameter
    @info "original parameter vector: $p_full"
    p_fix = Float64[]
    p_train = Float64[]
    fixed_indices = []
    train_indices = []
    for i in 1:length(p_full)
        if i in indexin(p_fixed, param_symbols)
            push!(p_fix, p_full[i])
            push!(fixed_indices, i)
        else
            push!(p_train, p_full[i])
            push!(train_indices, i)
        end
    end
    p_map = Vector{Int64}(undef, length(param_symbols))
    for (ix, i) in enumerate(fixed_indices)
        p_map[i] = ix
    end
    for (ix, i) in enumerate(train_indices)
        p_map[i] = ix + length(fixed_indices)
    end
    @info "remapped parameter vector: $(vcat(p_fix, p_train)[p_map])"
    return p_fix, p_train, p_map
end

function _initialize_params(p_fixed::Vector{Symbol}, p_full, surrogate::ZIP)
    param_symbols = ordered_param_symbols(surrogate)
    @assert !(nothing in indexin(p_fixed, param_symbols))  #ensure given p_fixed is a valid parameter
    @info "original parameter vector: $p_full"
    p_fix = Float64[]
    p_train = Float64[]
    fixed_indices = []
    train_indices = []
    for i in 1:length(p_full)
        if i in indexin(p_fixed, param_symbols)
            push!(p_fix, p_full[i])
            push!(fixed_indices, i)
        else
            push!(p_train, p_full[i])
            push!(train_indices, i)
        end
    end
    p_map = Vector{Int64}(undef, length(param_symbols))
    for (ix, i) in enumerate(fixed_indices)
        p_map[i] = ix
    end
    for (ix, i) in enumerate(train_indices)
        p_map[i] = ix + length(fixed_indices)
    end
    @info "remapped parameter vector: $(vcat(p_fix, p_train)[p_map])"
    return p_fix, p_train, p_map
end

function _initialize_params(p_fixed::Vector{Symbol}, p_full, surrogate::MultiDevice)
    param_symbols = Symbol[]
    for (i, s) in enumerate(vcat(surrogate.static_devices, surrogate.dynamic_devices))
        push!(param_symbols, Symbol("P_fraction_", i))
        push!(param_symbols, Symbol("Q_fraction_", i))
    end
    for s in surrogate.static_devices
        param_symbols = vcat(param_symbols, ordered_param_symbols(s))
    end
    for d in surrogate.dynamic_devices
        param_symbols = vcat(param_symbols, ordered_param_symbols(d))
    end

    @assert length(param_symbols) == length(unique(param_symbols))  #all parameters should be unique
    @assert !(nothing in indexin(p_fixed, param_symbols))  #ensure given p_fixed is a valid parameter
    @info "original parameter vector: $p_full"
    p_fix = Float64[]
    p_train = Float64[]
    fixed_indices = []
    train_indices = []
    for i in 1:length(p_full)
        if i in indexin(p_fixed, param_symbols)
            push!(p_fix, p_full[i])
            push!(fixed_indices, i)
        else
            push!(p_train, p_full[i])
            push!(train_indices, i)
        end
    end
    p_map = Vector{Int64}(undef, length(param_symbols))
    for (ix, i) in enumerate(fixed_indices)
        p_map[i] = ix
    end
    for (ix, i) in enumerate(train_indices)
        p_map[i] = ix + length(fixed_indices)
    end
    @info "remapped parameter vector: $(vcat(p_fix, p_train)[p_map])"
    return p_fix, p_train, p_map
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

function _initialize_training_output_dict(
    ::Union{PSIDS.SteadyStateNODEObsParams, PSIDS.SteadyStateNODEParams},
)
    return Dict{String, Any}(
        "loss" => DataFrames.DataFrame(
            Loss_initialization = Float64[],
            Loss_dynamic = Float64[],
            Loss = Float64[],
            iteration_time_seconds = Float64[],
            reached_ss = Bool[],
        ),
        "predictions" => DataFrames.DataFrame(
            trainable_parameters = Vector{Any}[],
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
        "chosen_iteration" => 0,
        "recorded_iterations" => [],
        "final_loss" => Dict{String, Vector{Float64}}(),
        "timing_stats" => [],
        "n_params_surrogate" => 0,
        "train_id" => "",
    )
end

function _initialize_training_output_dict(
    ::Union{
        PSIDS.ClassicGenParams,
        PSIDS.GFLParams,
        PSIDS.GFMParams,
        PSIDS.ZIPParams,
        PSIDS.MultiDeviceParams,
    },
)
    return Dict{String, Any}(
        "loss" => DataFrames.DataFrame(
            Loss_initialization = Float64[],
            Loss_dynamic = Float64[],
            Loss = Float64[],
            iteration_time_seconds = Float64[],
            reached_ss = Bool[],
        ),
        "predictions" => DataFrames.DataFrame(
            trainable_parameters = Vector{Any}[],
            parameters = Vector{Any}[],
            surrogate_solution = PhysicalModel_solution[],
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
        "chosen_iteration" => 0,
        "recorded_iterations" => [],
        "final_loss" => Dict{String, Vector{Float64}}(),
        "timing_stats" => [],
        "n_params_surrogate" => 0,
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
