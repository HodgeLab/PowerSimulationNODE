

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
