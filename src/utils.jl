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
