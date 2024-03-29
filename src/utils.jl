
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

function node_run_powerflow!(inputs...)
    logger = PSY.configure_logging(
        console_level = PSY_CONSOLE_LEVEL,
        file_level = PSY_FILE_LEVEL,
    )
    try
        Logging.with_logger(logger) do
            PowerFlows.solve_powerflow!(inputs...)
            return
        end
    finally
        close(logger)
        PSY.configure_logging(
            console_level = NODE_CONSOLE_LEVEL,
            file_level = NODE_FILE_LEVEL,
        )
    end
end

function build_grid_search!(base_option::TrainParams, args...)
    grid_dimensions = [length(a[2]) for a in args]
    grid_options = prod(grid_dimensions)
    #Note: tried train_params = repeat(base_option, grid_options) but problem with references.
    train_params = Vector{TrainParams}(undef, grid_options)
    for i in 1:length(train_params)
        train_params[i] = deepcopy(base_option)
    end

    dims_tuple = tuple(grid_dimensions...)
    iterator = CartesianIndices(dims_tuple)
    for (ix_outer, i) in enumerate(iterator)
        train_params[ix_outer].train_id = lpad(string(ix_outer), 3, "0")
        #When you change the train_id, also change the path where the modified validation system is stored
        mod_path = train_params[ix_outer].modified_surrogate_system_path
        d = dirname(mod_path)
        b = basename(mod_path)
        new_b = string(
            "modified_validation_system_",
            train_params[ix_outer].train_id,
            ".",
            split(b, ".")[2],
        )
        new_path = joinpath(d, new_b)
        train_params[ix_outer].modified_surrogate_system_path = new_path
        for (ix_inner, j) in enumerate(Tuple(i))
            _set_value!(train_params[ix_outer], args[ix_inner][1], args[ix_inner][2][j])
        end
    end
    return train_params
end

function build_random_search!(base_option::TrainParams, total_runs::Int64, args...)
    train_params = Vector{TrainParams}(undef, total_runs)
    for i in 1:total_runs
        train_params[i] = deepcopy(base_option)
    end
    for (i, tp) in enumerate(train_params)
        train_params[i].train_id = lpad(string(i), 3, "0")
        #When you change the train_id, also change the path where the modified validation system is stored
        mod_path = train_params[i].modified_surrogate_system_path
        d = dirname(mod_path)
        b = basename(mod_path)
        new_b = string(
            "modified_validation_system_",
            train_params[i].train_id,
            ".",
            split(b, ".")[2],
        )
        new_path = joinpath(d, new_b)
        train_params[i].modified_surrogate_system_path = new_path
        for a in args
            min_value = a[2].min
            max_value = a[2].max
            type = typeof(a[2].min)
            if haskey(a[2], :set)
                rand_value = rand(a[2].set)
            elseif type == Int64
                rand_value = rand(min_value:max_value)
            elseif type == Float64
                rand_value = min_value + (max_value - min_value) * rand()
            end
            _set_value!(tp, a[1], rand_value)
        end
    end
    return train_params
end

function _set_value!(TP, key, value)
    #CASE WHERE KEY IS ONE OF THE FIELDNAMES OF TrainParams itself 
    if key in fieldnames(TrainParams)
        for field_name in fieldnames(TrainParams)
            if key == field_name
                setfield!(TP, field_name, value)
                return
            end
        end
    end
    #CASE WHERE KEY IS A KEY IN THE NAMED TUPLE THAT SETS THE OPTIMIZERS 
    #NOTE - changes the parameter in all of the optimizer stages (if more than one)
    if key in [
        :sensealg,
        :algorithm,
        :log_η,
        :initial_stepnorm,
        :maxiters,
        :dynamic_solver,
        :steadystate_solver,
        :lb_loss,
        :curriculum,
        :curriculum_timespans,
        :fix_params,
        :loss_function,
        :α,
        :β,
        :residual_penalty,
    ]
        new_optimizer = []
        for entry in TP.optimizer
            if key in [:α, :β, :residual_penalty]
                new_value = merge(entry.loss_function, [key => value])
                new_key = :loss_function
                push!(new_optimizer, merge(entry, [new_key => new_value]))
            else
                push!(new_optimizer, merge(entry, [key => value]))
            end
        end
        TP.optimizer = new_optimizer
        return
    end

    #CASE WHERE THE KEY IS IN THE MODEL TYPE STRUCT 
    if key in fieldnames(typeof(TP.model_params))
        for field_name in fieldnames(typeof(TP.model_params))
            if key == field_name
                setfield!(TP.model_params, field_name, value)
                return
            end
        end
    end
    @error "KEY NOT FOUND, NO CHANGES MADE, ADD THIS CASE TO _set_value!()"
end
