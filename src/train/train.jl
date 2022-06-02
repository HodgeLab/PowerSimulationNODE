
function _build_exogenous_input_functions(sys, fault_data, branch_order)
    exs = []
    for train_data in fault_data
        V_funcs = []
        for (i, branch_name) in enumerate(branch_order)
            pvs_name = string(i, "_", branch_name)
            pvs = PSY.get_component(PSY.PeriodicVariableSource, sys, pvs_name)
            @assert !(pvs === nothing)
            Vm, Vθ = Source_to_function_of_time(pvs)
            push!(V_funcs, Vm)
            push!(V_funcs, Vθ)
        end

        R = train_data.connecting_impedance[:, 1]
        X = train_data.connecting_impedance[:, 2]
        RX = collect(Iterators.flatten(zip(R, X)))
        ex = generate_exogenous_input(V_funcs, RX)
        push!(exs, ex)
    end
    return exs
end
"""
    train(params::TrainParams)

Executes training according to params. Assumes the existence of the necessary input files. 

"""
function train(params::TrainParams)
    Random.seed!(params.rng_seed)
    #READ INPUT DATA AND SYSTEM
    sys = node_load_system(joinpath(params.input_data_path, "system.json"))
    SurrogateTrainInputs =
        Serialization.deserialize(joinpath(params.input_data_path, "data"))
    fault_data = SurrogateTrainInputs.train_data
    branch_order = SurrogateTrainInputs.branch_order
    n_ports = length(branch_order)

    #INSTANTIATE 
    surrogate = instantiate_surrogate(params, n_ports)
    optimizer = instantiate_optimizer(params)
    !(params.optimizer.adjust == "nothing") &&
        (optimizer_adjust = instantiate_optimizer_adjust(params))

    p_nn_init, _ = Flux.destructure(surrogate)
    @warn "# of parameters in surrogate", length(p_nn_init)
    res = nothing

    output = _initialize_training_output_dict()      #TODO - re-examine the output of data - better way to save flux model mid train and restart? 
    output["train_id"] = params.train_id
    output["n_params_surrogate"] = length(p_nn_init)

    exs = _build_exogenous_input_functions(sys, fault_data, branch_order)

    fault_indices = collect(1:length(fault_data))
    train_details = _add_curriculum_timespans_data!(fault_data, params.curriculum_timespans) #TODO - write for more than one curriculum timespan
    @assert length(train_details) == length(fault_data) == length(exs)
    train_groups = _generate_training_groups(fault_indices, params.curriculum)
    per_solve_maxiters = _calculate_per_solve_maxiters(
        params,
        length(train_groups) * length(train_groups[1]),
    )
    θ = p_nn_init
    total_time = @elapsed begin
        for group in train_groups
            res, output = _train(
                θ,
                surrogate,          #The acutal surrogate model
                fault_data,         #The dictionary with all the data for each type of fault including ground truth, ex(t), etc. # do we need branch order? don't think so 
                branch_order,
                exs,
                train_details,
                params,             #High level parameters (check if used...)
                optimizer,          #Optimizer for training (ADAM)
                group,              #The group of fault_indices to train on
                per_solve_maxiters, #per_solve_maxiters, #Maxiters for this training session 
                output,
            )
            θ = res.u #Update new minimum theta 
        end
    end
    output["total_time"] = total_time
    @info "End of training, calculating final loss for comparison:"
    final_loss_for_comparison = _calculate_final_loss(
        θ,
        surrogate,
        fault_data,
        branch_order,
        exs,
        train_details,
        params,
        train_groups[end],
        output,
    )

    output["final_loss"] = final_loss_for_comparison    #Do we need a final loss? Don't think so...
    if params.force_gc == true
        GC.gc()     #Run garbage collector manually before file write.
        @warn "FORCE GC!"
    end
    _capture_output(output, params.output_data_path, params.train_id)
    return true
    #TODO - uncomment try catch after debugging 
    #catch
    #    return false
    #end
end


function _train(
    θ::Vector{Float32},
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
    params::TrainParams,             #High level parameters (check if used...)
    optimizer::Union{Flux.Optimise.AbstractOptimiser, Optim.AbstractOptimizer},          #Optimizer for training (ADAM)
    group::Vector{Int64},              #The group of fault_indices to train on
    per_solve_maxiters::Int, #Maxiters for this training session 
    output::Dict{String, Any},  #The output dictionary to fill up 
)
    train_loader =
        Flux.Data.DataLoader(group; batchsize = 1, shuffle = true, partial = true)

    outer_loss_function = instantiate_outer_loss_function(
        surrogate,
        fault_data,
        branch_order,
        exs,
        train_details,
        params,
    )

    sensealg = instantiate_sensealg(params.optimizer)
    optfun = GalacticOptim.OptimizationFunction(
        (θ, P, fault_index) -> outer_loss_function(θ, fault_index),
        sensealg,
    )
    optprob = GalacticOptim.OptimizationProblem(optfun, θ)

    cb = instantiate_cb!(
    output,
    params.lb_loss,
    params.output_mode_skip,
    )
    #loss, lossA, lossB, lossC, surrogate_solution, fault_index = outer_loss_function(θ, 1)

    @warn "Everything instantiated, starting solve with one iteration"
    timing_stats_compile = @timed GalacticOptim.solve(
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

    @warn "Starting full train.", per_solve_maxiters
    timing_stats = @timed GalacticOptim.solve(
        optprob,
        optimizer,
        IterTools.ncycle(train_loader, per_solve_maxiters),
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

function _calculate_final_loss(
    θ::Vector{Float32},
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
    group::Vector{Int64},
    output::Dict{String, Any},
)
    outer_loss_function = instantiate_outer_loss_function(
        surrogate,
        fault_data,
        branch_order,
        exs,
        train_details,
        params,
    )

    lA_total = 0.0
    lB_total = 0.0
    lC_total = 0.0
    l_total = 0.0

    for fault_index in group
        loss, lossA, lossB, lossC, surrogate_solution, fault_index =
            outer_loss_function(θ, fault_index)
        lA_total += lossA
        lB_total += lossB
        lC_total += lossC
        l_total += loss
    end

    return l_total, lA_total, lB_total, lC_total
end


function _add_curriculum_timespans_data!(fault_data, curriculum_timespans)
    n_faults = length(fault_data)
    n_curriculums = length(curriculum_timespans)
    if n_curriculums == 1
        return repeat(curriculum_timespans, n_faults)
    end
end

function _generate_training_groups(dataset, curriculum)
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
            LossA = Float64[],
            LossB = Float64[],
            LossC = Float64[],
            Loss = Float64[],
        ),
        "predictions" => DataFrames.DataFrame(
            parameters = Vector{Any}[],
            surrogate_solution =  SteadyStateNeuralODE_solution[],
            fault_index = Int64[],
        ),
        "total_time" => [],
        "total_iterations" => 0,
        "recorded_iterations" => [],
        "final_loss" => [],
        "timing_stats_compile" => [],
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

function _calculate_per_solve_maxiters(params, n_groups)
    total_maxiters = params.maxiters
    per_solve_maxiters = Int(floor(total_maxiters / n_groups))
    if per_solve_maxiters == 0
        @error "The calculated maxiters per training group is 0. Adjust maxiters, the curriculum, or the size of the training dataset."
    end
    return per_solve_maxiters
end
