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
            V_funcs[i](t) + (y[i] * RX[i] - y[i + 1] * RX[i + 1])
        else
            V_funcs[i](t) + (y[i - 1] * RX[i] + y[i] * RX[i - 1])
        end for i in 1:length(V_funcs)
    ]
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

"""
    function build_params_list!(params_data, no_change_params, change_params)

Generate an array of `TrainParams` by combinatorally iterating over `change_params`. If a parameter is excluded from `change_params` and `no_change_params` the default value is used. 
"""
function build_params_list!(params_data, no_change_params, change_params)
    train_id = 1
    starting_dict = no_change_params
    dims = []
    for (k, v) in change_params
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
