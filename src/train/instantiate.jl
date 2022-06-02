function optimizer_map(key)
    d = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)
    return d[key]
end

function steadystate_solver_map(solver, tols)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => OrdinaryDiffEq.Tsit5,
    )
    return SteadyStateDiffEq.DynamicSS(d[solver](), abstol = tols[1], reltol = tols[2])
end

function solver_map(key)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => OrdinaryDiffEq.Tsit5,
    )
    return d[key]
end

function instantiate_steadystate_solver(inputs)
    return steadystate_solver_map(inputs.solver, inputs.tols)
end

function solver_sensealg_map(key)
    d = Dict(
        "InterpolatingAdjoint" =>
            DiffEqSensitivity.InterpolatingAdjoint(checkpointing = false),
        "InterpolatingAdjoint_checkpointing" =>
            DiffEqSensitivity.InterpolatingAdjoint(checkpointing = true),
    )
    return d[key]
end

function observation_map(key)
    d = Dict("first_two" => ((x, θ) -> x[1:2, :], Float64[]))
    return d[key]
end

function sensealg_map(key)
    d = Dict(
        "ForwardDiff" => GalacticOptim.AutoForwardDiff,
        "Zygote" => GalacticOptim.AutoZygote,
    )
    return d[key]
end

function activation_map(key)
    d = Dict("relu" => Flux.relu, "hardtanh" => Flux.hardtanh, "sigmoid" => Flux.sigmoid)
    return d[key]
end

function instantiate_solver_sensealg(inputs)
    return solver_sensealg_map(inputs.sensealg)
end

function instantiate_solver(inputs)
    return solver_map(inputs.solver)()
end

function instantiate_observation(inputs)
    return observation_map(inputs.observation_function)
end

function instantiate_sensealg(inputs)
    return sensealg_map(inputs.sensealg)()
end

function instantiate_optimizer(inputs)
    if inputs.optimizer.primary == "Adam"
        return optimizer_map(inputs.optimizer.primary)(inputs.optimizer.primary_η)
    elseif inputs.optimizer.primary == "Bfgs"
        return optimizer_map(inputs.optimizer.primary)()
    end
end

function instantiate_optimizer_adjust(inputs)
    if inputs.optimizer.adjust == "Adam"
        return optimizer_map(inputs.optimizer.adjust)(inputs.optimizer_adjust_η)
    elseif inputs.optimizer.adjust == "Bfgs"
        return optimizer_map(inputs.optimizer.adjust)()
    end
end

function instantiate_surrogate(TrainParams, n_ports)
    steadystate_sensealg = instantiate_solver_sensealg(TrainParams.steady_state_solver)
    steadystate_solver = instantiate_steadystate_solver(TrainParams.steady_state_solver)
    dynamic_sensealg = instantiate_solver_sensealg(TrainParams.dynamic_solver)
    dynamic_solver = instantiate_solver(TrainParams.dynamic_solver)
    model_initializer = _instantiate_model_initializer(TrainParams, n_ports)
    model_node = _instantiate_model_node(TrainParams, n_ports)
    model_observation = _instantiate_model_observation(TrainParams, n_ports)
    display(model_initializer)
    display(model_node)
    display(model_observation)

    dynamic_reltol = TrainParams.dynamic_solver.tols[1]
    dynamic_abstol = TrainParams.dynamic_solver.tols[2] #TODO check order (change to named tuple?)
    dynamic_maxiters = TrainParams.dynamic_solver.maxiters

    return SteadyStateNeuralODE(
        model_initializer,
        model_node,
        model_observation,
        steadystate_solver,
        dynamic_solver;
        abstol = dynamic_abstol,
        reltol = dynamic_reltol,
        maxiters = dynamic_maxiters,
        #  sensealg = dynamic_sensealg,   #get rid of sensealg, see if speed up 
    )
    #TODO - cleaner implementation of dealing with kwargs... (split kwargs between dynamic and ss solvers? )
end

function _instantiate_model_initializer(TrainParams, n_ports)
    initializer_params = TrainParams.model_initializer
    hidden_states = TrainParams.hidden_states
    type = initializer_params.type
    n_layer = initializer_params.n_layer
    width_layers = initializer_params.width_layers
    activation = activation_map(initializer_params.activation)
    normalization = initializer_params.normalization    #TODO built into the first layer? 
    vector_layers = []
    if type == "dense"
        push!(vector_layers, Dense(4 * n_ports, width_layers, activation))   #first layer  TODO make 4 a constant 
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation)) #hidden layers 
        end
        push!(vector_layers, Dense(width_layers, hidden_states, activation))
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    elseif type == "OutputParams"
        @error "OutputParams layer for inititalizer not yet implemented"
    end
    return model
end

function _instantiate_model_node(TrainParams, n_ports)
    node_params = TrainParams.model_node
    hidden_states = TrainParams.hidden_states
    type = node_params.type
    n_layer = node_params.n_layer
    width_layers = node_params.width_layers
    activation = activation_map(node_params.activation)
    normalization = node_params.normalization    #TODO built into the first layer? 
    vector_layers = []
    if type == "dense"
        push!(
            vector_layers,
            Parallel(
                +,
                Dense(2 * n_ports, hidden_states, activation),
                Dense(hidden_states, hidden_states, activation),
            ),
        ) #TODO make 2 a constant 
        if n_layer >= 1
            push!(vector_layers, Dense(hidden_states, width_layers, activation))
        end
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation)) #hidden layers 
        end
        push!(vector_layers, Dense(width_layers, hidden_states, activation))
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    end
    return model
end

function _instantiate_model_observation(TrainParams, n_ports)
    observation_params = TrainParams.model_observation
    hidden_states = TrainParams.hidden_states
    type = observation_params.type
    n_layer = observation_params.n_layer
    width_layers = observation_params.width_layers
    activation = activation_map(observation_params.activation)
    normalization = observation_params.normalization    #TODO built into the first layer? 
    vector_layers = []
    if type == "dense"
        push!(vector_layers, Dense(hidden_states, width_layers, activation))
        for i in 1:n_layer
            push!(vector_layers, Dense(width_layers, width_layers, activation)) #hidden layers 
        end
        push!(vector_layers, Dense(width_layers, 2 * n_ports, activation))   #make 2 a constant 
        tuple_layers = Tuple(x for x in vector_layers)
        model = Chain(tuple_layers)
    end
    return model
end

function _inner_loss_function(surrogate_solution, ground_truth_subset, params)
    rmse_weight = params.loss_function.type_weights.rmse
    mae_weight = params.loss_function.type_weights.mae
    A_weight = params.loss_function.component_weights.A
    B_weight = params.loss_function.component_weights.B
    C_weight = params.loss_function.component_weights.C
    scale = params.loss_function.scale      #TODO  - add in the scaling based on the ground truth values (for B and C ) 
    r0_pred = surrogate_solution.r0_pred        #or scratch this idea, might not make sense to change the loss for every training set, etc. 
    r0 = surrogate_solution.r0
    i_series = surrogate_solution.i_series
    ϵ = surrogate_solution.ϵ

    lossA =
        A_weight * (mae_weight * mae(r0_pred, r0) + rmse_weight * sqrt(mse(r0_pred, r0)))
    if size(ground_truth_subset) == size(i_series)
        lossB =
            B_weight * (
                mae_weight * mae(ground_truth_subset, i_series) +
                rmse_weight * sqrt(mse(ground_truth_subset, i_series))
            )
    else
        lossB = Inf
    end

    lossC =
        C_weight * (
            mae_weight * mae(ϵ, zeros(length(ϵ))) +
            rmse_weight * sqrt(mse(ϵ, zeros(length(ϵ))))
        )
    return lossA, lossB, lossC
end

function instantiate_outer_loss_function(
    surrogate::SteadyStateNeuralODE,
    fault_data::Array{TrainData},
    branch_order::Array{String},
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
)
    return (θ, fault_index) -> _outer_loss_function(
        θ,
        fault_index,
        surrogate,
        fault_data,
        branch_order,
        exs,
        train_details,
        params,
    )
end

function _outer_loss_function(
    θ,
    fault_index::Int,
    surrogate::SteadyStateNeuralODE,
    fault_data::Array{TrainData},
    branch_order::Array{String},
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
)
    ex = exs[fault_index]
    powerflow = fault_data[fault_index].powerflow
    tsteps = fault_data[fault_index].tsteps
    groundtruth_current = fault_data[fault_index].groundtruth_current
    index_subset = _find_subset(tsteps, train_details[fault_index])   # #TODO - implement function to find index_subset based on tspan and batching factor
    t_subset = tsteps[index_subset]
    groundtruth_subset = groundtruth_current[:, index_subset]    #Todo - use views, time / compare anser? 
    surrogate_solution = surrogate(ex, powerflow, t_subset, θ)
    lossA, lossB, lossC =
        _inner_loss_function(surrogate_solution, groundtruth_subset, params)
    return lossA + lossB + lossC, lossA, lossB, lossC, surrogate_solution, fault_index
end

function _find_subset(tsteps, train_details)
    return trues(length(tsteps))
end

function instantiate_simple_cb(fault_data, exs, ls, lAs, lBs, lCs)
    return (θ, l, lA, lB, lC, surrogate_solution, fault_index) -> _simple_cb(
        θ,
        l,
        lA,
        lB,
        lC,
        ls,
        lAs,
        lBs,
        lCs,
        surrogate_solution,
        fault_index,
        fault_data,
        exs,
    )
end

function _simple_cb(
    θ,
    l,
    lA,
    lB,
    lC,
    ls,
    lAs,
    lBs,
    lCs,
    surrogate_solution,
    fault_index,
    fault_data,
    exs,
)
    push!(ls, l)
    push!(lAs, lA)
    push!(lBs, lB)
    push!(lCs, lC)
    println(fault_index)
    println(θ[1])
    println(l)
    p = plot_overview(surrogate_solution, fault_index, fault_data, exs)
    display(p)
    return false
end

function instantiate_cb!(
    output,
    lb_loss,
    exportmode_skip,
)
    if Sys.iswindows() || Sys.isapple()
        print_loss = true
    else
        print_loss = false
    end


    return (p, l, lA, lB, lC,  surrogate_solution, fault_index) -> _cb!(
        p,
        l,
        lA,
        lB,
        lC,
        surrogate_solution,
        fault_index,
        output, 
        lb_loss,
        print_loss,
        exportmode_skip,
    )

end


function _cb!(
    p,
    l,
    lA,
    lB,
    lC,
    surrogate_solution,
    fault_index,
    output, 
    lb_loss,
    print_loss,
    exportmode_skip,
)
    
    push!(output["loss"], (lA, lB, lC, l))
    output["total_iterations"] += 1
    if mod(output["total_iterations"], exportmode_skip) == 0
     
        push!(output["predictions"], ([p], surrogate_solution, fault_index))
        push!(output["recorded_iterations"], output["total_iterations"]) #Indices of the recorded iterations... 
    end

    if (print_loss)
        println(l)
    end
    (l > lb_loss) && return false
    return true
end