
function _build_exogenous_input_functions(
    train_data_params::NamedTuple{
        (:operating_points, :perturbations, :params, :system),
        Tuple{
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
            String,
        },
    },
    train_dataset::Vector{PSIDS.SurrogateDataset},
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
                )   #todo - implement dispatch for each type of perturbation
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
    D <: PSIDS.SurrogateDataset,
}
    @warn typeof(data)
    #@assert length(data.branch_order) == length(perturbation)
    V_funcs = []
    if train_data_params.system == "full"
        @warn "Generating train data from full system"
        for i in 1:size(data.groundtruth_voltage)[1]
            function V(t)
                ix_after = findfirst(x -> x > t, data.tsteps)
                if ix_after === nothing  #correct behavior at end of timespan
                    ix_after = length(data.tsteps)
                end
                t_before = data.tsteps[ix_after - 1]
                t_after = data.tsteps[ix_after]
                val_before = data.groundtruth_voltage[i, ix_after - 1]
                val_after = data.groundtruth_voltage[i, ix_after]
                frac = (t - t_before) / (t_after - t_before)
                val = val_before + frac * (val_after - val_before)
                return val
            end
            push!(V_funcs, V)
        end

    elseif train_data_params.system == "reduced"
        for (ix, p_single) in enumerate(perturbation)
            Vm, Vθ = _single_perturbation_to_function_of_time(p_single, data, ix)
            push!(V_funcs, Vm)
            push!(V_funcs, Vθ)
        end
    else
        @error "invalid value"
    end
    R = data.connecting_impedance[:, 1]
    X = data.connecting_impedance[:, 2]
    RX = collect(Iterators.flatten(zip(R, X)))
    return generate_exogenous_input(V_funcs, RX)
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
    V_bias = data.powerflow[(port_ix * 4) - 1]
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
    θ_bias = data.powerflow[port_ix * 4]
    θ_freqs = single_perturbation.internal_voltage_frequencies
    θ_coeffs = single_perturbation.internal_voltage_coefficients
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

function _single_perturbation_to_function_of_time(
    single_perturbation::PSIDS.VStep,
    data::PSIDS.SteadyStateNODEData,
    port_ix::Int64,
)
    @warn "TODO - implement for VStep"
end

#TODO - cleaner way to set new parameters so you don't have to pass the Flux surrogate around
function evaluate_loss(sys, θ, groundtruth_dataset, params, connecting_branches, surrogate)
    for s in PSY.get_components(PSIDS.SteadyStateNODE, sys)
        PSIDS.set_initializer_parameters!(s, θ[1:(surrogate.len)])
        PSIDS.set_node_parameters!(
            s,
            θ[(surrogate.len + 1):(surrogate.len + surrogate.len2)],
        )
        PSIDS.set_observer_parameters!(s, θ[(surrogate.len + surrogate.len2 + 1):end])
        display(s)
    end

    operating_points = params.operating_points
    perturbations = params.perturbations
    generate_data_params = params.params
    @warn operating_points
    @warn perturbations
    @warn generate_data_params
    surrogate_dataset = PSIDS.generate_surrogate_data(
        sys,
        sys,
        perturbations,
        operating_points,
        PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        generate_data_params,
        dataset_aux = groundtruth_dataset,
    )
    #@warn surrogate_dataset
    #TODO- calculate the loss metrics by comparing surrogate_dataset and groundtruth_dataset.
    #Make sure this function can be used for the test system equivalently 

    p1 = Plots.plot(
        groundtruth_dataset[1].tsteps,
        groundtruth_dataset[1].groundtruth_current[1, :],
    )
    Plots.plot!(
        p1,
        surrogate_dataset[1].tsteps,
        surrogate_dataset[1].groundtruth_current[1, :],
    )
    p2 = Plots.plot(
        groundtruth_dataset[1].tsteps,
        groundtruth_dataset[1].groundtruth_current[2, :],
    )
    Plots.plot!(
        p2,
        surrogate_dataset[1].tsteps,
        surrogate_dataset[1].groundtruth_current[2, :],
    )
    display(Plots.plot(p1, p2))
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
    @info "Length of actual training dataset (stable conditions):", length(train_dataset)   #TODO - filter based on stable == true 
    @info "Length of possible validation conditions (number of fault/operating point combinations):",
    length(validation_dataset)
    @info "Length of actual validation dataset (stable conditions):",
    length(validation_dataset)  #TODO - filter based on stable == true 
    @info "Length of possible test conditions (number of fault/operating point combinations):",
    length(test_dataset)
    @info "Length of actual test dataset (stable conditions):", length(test_dataset)  #TODO - filter based on stable == true 

    n_ports = length(connecting_branches)
    @info "Surrogate contains $n_ports ports"

    #READ VALIDATION SYSTEM AND ADD SURROGATE COMPONENT WITH STRUCTURE BASED ON PARAMS
    sys_validation = node_load_system(params.surrogate_system_path)
    sources = collect(
        PSY.get_components(PSY.Source, sys_validation, x -> PSY.get_name(x) !== "InfBus"),
    )
    (length(sources) > 1) &&
        @error "Surrogate with multiple input/output ports not yet supported"
    psid_surrogate = instantiate_surrogate_psid(params, n_ports, PSY.get_name(sources[1]))
    PSY.add_component!(sys_validation, psid_surrogate, sources[1])
    display(sys_validation)

    #INSTANTIATE 
    surrogate = instantiate_surrogate_flux(params, n_ports)
    optimizer = instantiate_optimizer(params)
    !(params.optimizer.adjust == "nothing") &&
        (optimizer_adjust = instantiate_optimizer_adjust(params))

    p_nn_init, _ = Flux.destructure(surrogate)
    n_parameters = length(p_nn_init)
    @info "Surrogate has $n_parameters parameters"
    res = nothing

    output = _initialize_training_output_dict()
    output["train_id"] = params.train_id
    output["n_params_surrogate"] = n_parameters
    @warn size(train_dataset[1].powerflow)
    @warn size(train_dataset[1].groundtruth_current)
    exs = _build_exogenous_input_functions(params.train_data, train_dataset)    #can build ex from the components in params.train_data or from the dataset values by interpolating...

    @warn exs[1](1.0, [1, 1])
    #want to test how this impacts the speed of a single train iteration (the interpolation)
    train_details = params.curriculum_timespans
    fault_indices = collect(1:length(train_dataset))
    timespan_indices = collect(1:length(params.curriculum_timespans))
    @assert length(train_dataset) == length(exs)
    train_groups =
        _generate_training_groups(fault_indices, timespan_indices, params.curriculum)
    n_trains = length(train_groups) #TODO - doublecheck 
    per_solve_maxiters = _calculate_per_solve_maxiters(params, n_trains)
    @info "Curriculum pairings (fault_index, timespan_index)", train_groups
    @info "Based on size of train dataset and curriculum timespans parameter, will have $n_trains trains with a maximum of $per_solve_maxiters iterations per train"
    if isempty(params.p_start)
        θ = p_nn_init
    else
        θ = params.p_start
    end
    #try
    total_time = @elapsed begin
        for group in train_groups
            res, output = _train(
                θ,
                surrogate,
                connecting_branches,
                train_dataset,
                validation_dataset,
                sys_validation,
                exs,
                train_details,
                params,
                optimizer,
                group,
                per_solve_maxiters,
                output,
            )
            θ = res.u
        end
    end
    output["total_time"] = total_time
    @info "End of training, calculating final loss for comparison:"
    final_loss_for_comparison = _calculate_final_loss(
        θ,
        surrogate,
        test_dataset,
        sys_validation,
        exs,
        train_details,
        params,
        train_groups[end],
    )

    output["final_loss"] = final_loss_for_comparison
    if params.force_gc == true
        GC.gc()     #Run garbage collector manually before file write.
        @warn "FORCE GC!"
    end
    _capture_output(output, params.output_data_path, params.train_id)
    return true, θ
    #TODO - uncomment try catch after debugging 
    #catch
    #    return false, θ
    #end
end

function _train(
    θ::Vector{Float32},
    surrogate::SteadyStateNeuralODE,
    connecting_branches::Vector{Tuple{String, Symbol}},   #NEW 
    train_dataset::Vector{PSIDS.SurrogateDataset},
    validation_dataset::Vector{PSIDS.SurrogateDataset},
    sys_validation::PSY.System,
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
    optimizer::Union{Flux.Optimise.AbstractOptimiser, Optim.AbstractOptimizer},
    group::Vector{Tuple{Int64, Int64}},
    per_solve_maxiters::Int,
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
    optfun = GalacticOptim.OptimizationFunction(
        (θ, P, fault_timespan_index) -> outer_loss_function(θ, fault_timespan_index),
        sensealg,
    )
    optprob = GalacticOptim.OptimizationProblem(optfun, θ)

    cb = instantiate_cb!(
        output,
        params,
        validation_dataset,
        sys_validation,
        connecting_branches,
        surrogate,
    )

    loss, lossA, lossB, lossC, surrogate_solution, fault_index =
        outer_loss_function(θ, (1, 1))
    display(loss)
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
    test_dataset::Vector{PSIDS.SurrogateDataset},
    sys_validation::PSY.System, #TODO - calculate final loss by adding the surrogate to the validation system and testing against test_dataset
    exs,
    train_details::Vector{
        NamedTuple{
            (:tspan, :batching_sample_factor),
            Tuple{Tuple{Float64, Float64}, Float64},
        },
    },
    params::TrainParams,
    group::Vector{Tuple{Int64, Int64}},
)
    outer_loss_function =
        instantiate_outer_loss_function(surrogate, test_dataset, exs, train_details, params)

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
            LossA = Float64[],
            LossB = Float64[],
            LossC = Float64[],
            Loss = Float64[],
        ),
        "predictions" => DataFrames.DataFrame(
            parameters = Vector{Any}[],
            surrogate_solution = SteadyStateNeuralODE_solution[],
            fault_index = Tuple{Int64, Int64}[],
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
