"""
    PVS_to_function_of_time(source::PeriodicVariableSource)

Takes in a PeriodicVariableSource from PowerSystems and generates functions of time for voltage magnitude and angle
"""
function Source_to_function_of_time(source::PSY.PeriodicVariableSource)
    V_bias = PSY.get_internal_voltage_bias(source)
    V_freqs = PSY.get_internal_voltage_frequencies(source)
    V_coeffs = PSY.get_internal_voltage_coefficients(source)
    function V(t)
        val = V_bias
        for (i, ω) in enumerate(V_freqs)
            val += V_coeffs[i][1] * sin.(ω * t)
            val += V_coeffs[i][2] * cos.(ω * t)
        end
        return val
    end
    θ_bias = PSY.get_internal_angle_bias(source)
    θ_freqs = PSY.get_internal_angle_frequencies(source)
    θ_coeffs = PSY.get_internal_angle_coefficients(source)
    function θ(t)
        val = θ_bias
        for (i, ω) in enumerate(θ_freqs)
            val += θ_coeffs[i][1] * sin.(ω * t)
            val += θ_coeffs[i][2] * cos.(ω * t)
        end
        return val
    end
    return (V, θ)
end

function generate_exogenous_input(V_funcs, RX)
    @assert length(V_funcs) == length(RX)
    return (t, y) -> _exogenous_input(t, y, V_funcs, RX)
end

function _exogenous_input(t, y, V_funcs, RX)
    return [
        if isodd(i)
            V_funcs[i](t) * cos(V_funcs[i + 1](t)) + (y[i] * RX[i] - y[i + 1] * RX[i + 1])
        else
            V_funcs[i - 1](t) * sin(V_funcs[i](t)) + (y[i - 1] * RX[i] + y[i] * RX[i - 1])
        end for i in 1:length(V_funcs)
    ]
end

function Source_to_function_of_time(source::PSY.Source)
    function V(t)
        return PSY.get_internal_voltage(source)
    end

    function θ(t)
        return PSY.get_internal_angle(source)
    end
    return (V, θ)
end

"""
    node_load_system(inputs...)

Configures logging and calls PSY.System(inputs...). 
"""
function node_load_system(inputs...)
    logger = PSY.configure_logging(
        console_level = PSY_CONSOLE_LEVEL,
        file_level = PSY_FILE_LEVEL,
    )
    try
        Logging.with_logger(logger) do
            sys = PSY.System(inputs...)
            return sys
        end
    finally
        close(logger)
        PSY.configure_logging(
            console_level = NODE_CONSOLE_LEVEL,
            file_level = NODE_FILE_LEVEL,
        )
    end
end

function build_params_list!(params_data, no_change_params, change_params)
    train_id = 1
    starting_dict = no_change_params
    dims = []
    for (k, v) in change_params
        @warn k
        @warn length(v)
        push!(dims, length(v))
    end
    dims_tuple = tuple(dims...)
    iterator = CartesianIndices(dims_tuple)
    for i in iterator
        for (j, (key, value)) in enumerate(change_params)
            starting_dict[key] = value[i[j]]
        end
        starting_dict[:train_id] = string(train_id)
        push!(params_data, TrainParams(; starting_dict...))
        starting_dict = no_change_params
        train_id += 1
    end
end

function build_training_group(training_group_dict)
    training_group = []
    for i in range(training_group_dict[:training_groups], 1, step = -1)
        tspan = (training_group_dict[:tspan][1], training_group_dict[:tspan][2] / i)
        shoot_times = filter(x -> x < tspan[2], training_group_dict[:shoot_times])
        multiple_shoot_continuity_term =
            training_group_dict[:multiple_shoot_continuity_term]
        batching_sample_factor = training_group_dict[:batching_sample_factor]
        push!(
            training_group,
            (
                tspan = tspan,
                shoot_times = shoot_times,
                multiple_shoot_continuity_term = multiple_shoot_continuity_term,
                batching_sample_factor = batching_sample_factor,
            ),
        )
    end
    return training_group
end

function build_training_groups_list(no_change_fields, change_fields)
    training_groups_list = []
    starting_dict = no_change_fields
    dims = []
    for (k, v) in change_fields
        @warn "training groups sub-category", k
        @warn length(v)
        push!(dims, length(v))
    end
    dims_tuple = tuple(dims...)
    iterator = CartesianIndices(dims_tuple)
    for i in iterator
        for (j, (key, value)) in enumerate(change_fields)
            starting_dict[key] = value[i[j]]
        end
        push!(training_groups_list, build_training_group(starting_dict))
    end

    return training_groups_list
end
