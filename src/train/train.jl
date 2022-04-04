function _calculate_loss_function_scaling(params, fault_data)
    if params.loss_function_scale == "range"
        dim_ground_truth = 0
        for (key, value) in fault_data
            dim_ground_truth = size(value[:ground_truth])[1]
            break
        end
        full_ground_truth = zeros(Float64, (dim_ground_truth, 0))
        for (key, value) in fault_data
            full_ground_truth = hcat(full_ground_truth, value[:ground_truth])
        end
        ground_truth_scale =
            maximum(full_ground_truth, dims = 2) - minimum(full_ground_truth, dims = 2)
    elseif params.loss_function_scale == "none"
        dim_ground_truth = 0
        for (key, value) in fault_data
            dim_ground_truth = size(value[:ground_truth])[1]
            break
        end
        ground_truth_scale = ones(Float64, (dim_ground_truth, 1))
    else
        @warn "Cannot determine loss function scaling"
    end
    return ground_truth_scale
end
function _turn_node_on(surr_prob_node_off, params, P)
    P.scale[2] = params.node_output_scale
    p = vectorize(P)
    surr_prob = OrdinaryDiffEq.remake(surr_prob_node_off, p = p)
    return surr_prob
end

function _initialize_surrogate(
    params,
    nn,
    nn_params,
    M,
    tsteps,
    fault_dict,
    surr,
    N_ALGEBRAIC_STATES,
    ODE_ORDER,
)
    #Determine Order of the Surrogate 
    order_surr = ODE_ORDER + 2 + params.node_unobserved_states + N_ALGEBRAIC_STATES    #2 states for the node current
    x₀_surr = zeros(order_surr)

    #Build Surrogate Vector 
    P = SurrParams()
    if params.ode_model == "none"
        if !(isempty(fault_dict[:p_ode]))
            @warn "ODE model is not present in surrogate, yet parameters for an ode are provided in train data. These parameters will be ignored"
        end
        P.ode = []
    else
        P.ode = fault_dict[:p_ode]
    end
    P.pf = fault_dict[:p_pf]
    P.network = fault_dict[:p_network]
    P.nn = nn_params # DiffEqFlux.initial_params(nn)
    if params.ode_model == "none"
        P.scale = [params.node_input_scale, params.node_output_scale]
        #x₀_surr[1:(end-2)] = 0.0  - initialized to zero already 
    else
        P.scale = [params.node_input_scale, 0.0]
        x₀ = fault_dict[:x₀]
        x₀_surr[(3 + params.node_unobserved_states + N_ALGEBRAIC_STATES):(ODE_ORDER + 2 + params.node_unobserved_states + N_ALGEBRAIC_STATES)] =
            x₀
    end
    p = vectorize(P)
    P_pf, Q_pf, V_pf, θ_pf = P.pf
    Vr_pf = V_pf * cos(θ_pf)
    Vi_pf = V_pf * sin(θ_pf)
    Ir_pf = (P_pf * Vr_pf + Q_pf * Vi_pf) / (Vr_pf^2 + Vi_pf^2)
    Ii_pf = (P_pf * Vi_pf - Q_pf * Vr_pf) / (Vi_pf^2 + Vr_pf^2)

    x₀_surr[1:2] = [Ir_pf, Ii_pf]

    if params.ode_model != "none"   #only do an NL solve if there is an ODE part. Otherwise, set initial conditions and try to solve.
        @warn "Power flow results", P.pf
        h = _get_init_surr(p, Ir_pf, Ii_pf, surr)
        res_surr = NLsolve.nlsolve(h, x₀_surr)
        @assert NLsolve.converged(res_surr)
        dx = similar(x₀_surr)
        surr(dx, res_surr.zero, p, 0.0)
        @assert all(isapprox.(dx, 0.0; atol = 1e-8))
        x₀_surr = res_surr.zero
    end
    @info "x0 surr", x₀_surr
    tspan = (tsteps[1], tsteps[end])
    surr_func = OrdinaryDiffEq.ODEFunction(surr, mass_matrix = M)
    surr_prob = OrdinaryDiffEq.ODEProblem(surr_func, x₀_surr, tspan, p)
    @info "states in surrogate", length(x₀_surr)
    return surr_prob, P
end

function _verify_psid_node_off(surr_prob, params, solver, tsteps, fault_dict)
    i_ver = vcat(Float64.(fault_dict[:ir_node_off])', Float64.(fault_dict[:ii_node_off])')
    sol = OrdinaryDiffEq.solve(
        surr_prob,
        solver,
        abstol = params.solver_tols[1],
        reltol = params.solver_tols[2],
        saveat = tsteps,
    )
    #=
    NOTE: Use to visualize the verification in case @assert fails
    p1 = Plots.plot(i_ver[1,:], label = "provided verification data from psid")
    Plots.plot!(p1, sol[1,:], label = "Ir_out")
    Plots.plot!(p1,sol[3,:], label = "Ir_nn")
    Plots.plot!(p1, sol[9,:], label = "Ir_filter")
    Plots.png(p1 ,"test_verify")
    =#
    @assert mae(sol[1, :], i_ver[1, :]) < 1e-5
    @assert mae(sol[2, :], i_ver[2, :]) < 1e-5
end

function _calculate_final_loss(
    params,
    θ,
    solver,
    solver_tols,
    solver_sensealg,
    pvs_names,
    fault_data,
    tsteps,
    ground_truth_scale,
    output,
    observation_function,
)
    θ_vec, θ_lengths = combine_θ(θ)
    inner_loss_function =
        instantiate_inner_loss_function(params.loss_function_weights, ground_truth_scale)

    ground_truth = concatonate_ground_truth(fault_data, pvs_names, :)
    t_current = concatonate_t(tsteps, pvs_names, :)

    pvs_ranges = generate_pvs_ranges(pvs_names, length(tsteps))

    outer_loss_function = instantiate_outer_loss_function(
        solver,
        solver_tols,
        solver_sensealg,
        fault_data,
        inner_loss_function,
        (
            shoot_times = [],
            multiple_shoot_continuity_term = (100.0, 100.0), #Shouldn't matter, single shoot
            batching_sample_factor = 1.0,
        ),
        θ_lengths,
        params,
        observation_function,
        pvs_names,
        pvs_ranges,
    )

    cb = instantiate_cb!(
        output,
        params.lb_loss,
        params.output_mode,
        1,  #Don't want to skip the final callback
        0,  #range_count
        pvs_names,
    )
    final_loss_for_comparison = outer_loss_function(θ_vec, ground_truth, t_current)

    cb(
        θ_vec,
        final_loss_for_comparison[1],
        final_loss_for_comparison[2],
        final_loss_for_comparison[3],
        final_loss_for_comparison[4],
    )
    return final_loss_for_comparison[1]
end

function _get_init_surr(p, Ir_pf, Ii_pf, surr)
    return (dx, x) -> begin
        dx[1] = 0
        x[1] = Ir_pf
        dx[2] = 0
        x[2] = Ii_pf
        surr(dx, x, p, 0.0)
    end
end

function _calculate_per_solve_maxiters(params, n_faults)
    total_maxiters = params.maxiters
    groupsize_faults = params.groupsize_faults
    n_groups = length(params.training_groups)
    if rem(n_faults, groupsize_faults) != 0
        @error "number of faults not divisible by groupsize_faults parameter!"
    end
    per_solve_maxiters =
        Int(floor((total_maxiters * groupsize_faults) / (n_faults * n_groups)))
    @info "per solve maxiters:", per_solve_maxiters
    if per_solve_maxiters == 0
        @error "maxiters is too low. The calculated maxiters per solve is 0! cannot train"
    end
    return per_solve_maxiters
end

"""
    train(params::NODETrainParams)

Executes training according to params. Assumes the existence of the necessary input files. 

"""
function train(params::NODETrainParams)
    @info "TRAIN!!!"
    #READ INPUT DATA AND SYSTEM
    sys = node_load_system(joinpath(params.input_data_path, "system.json"))
    TrainInputs = Serialization.deserialize(joinpath(params.input_data_path, "data"))
    n_observable_states = TrainInputs.n_observable_states
    tsteps = TrainInputs.tsteps
    fault_data = TrainInputs.fault_data

    #INSTANTIATE
    sensealg = instantiate_sensealg(params)
    solver = instantiate_solver(params)
    solver_tols = params.solver_tols
    solver_sensealg = instantiate_solver_sensealg(params)
    optimizer = instantiate_optimizer(params)
    nn_full = instantiate_nn(params, n_observable_states)
    p_nn_init, nn = Flux.destructure(nn_full)
    M = instantiate_M(params)

    observation_function, observation_params = instantiate_observation(params)  #observation_function(vector,params)

    !(params.optimizer_adjust == "nothing") &&
        (optimizer_adjust = instantiate_optimizer_adjust(params))

    pvss = collect(PSY.get_components(PSY.PeriodicVariableSource, sys))
    ground_truth_scale = _calculate_loss_function_scaling(params, fault_data)
    res = nothing
    output = Dict{String, Any}(
        "loss" => DataFrames.DataFrame(
            PVS_name = Vector{String}[],
            RangeCount = Int[],
            Loss = Float64[],
        ),
        "parameters" => DataFrames.DataFrame(Parameters = Vector{Any}[]),
        "predictions" => DataFrames.DataFrame(
            t_prediction = Vector{Any}[],
            prediction = Vector{Any}[],
            observation = Vector{Any}[],
        ),
        "total_time" => [],
        "total_iterations" => 0,
        "recorded_iterations" => [],
        "final_loss" => [],
        "timing_stats_compile" => [],
        "timing_stats" => [],
        "n_params_nn" => length(p_nn_init),
        "train_id" => params.train_id,
    )
    per_solve_maxiters = _calculate_per_solve_maxiters(params, length(pvss))

    !(params.learn_initial_condition_unobserved_states) &&
        (unobserved_u0 = rand(params.node_unobserved_states))

    for pvs in pvss
        @warn "PVS:", PSY.get_name(pvs)
        Vm, Vθ = Source_to_function_of_time(pvs)
        fault_dict = fault_data[PSY.get_name(pvs)]
        psid_results_object = fault_dict[:psid_results_object]
        surr, N_ALGEBRAIC_STATES, ODE_ORDER = instantiate_surr(
            params,
            nn,
            p_nn_init,
            n_observable_states,
            Vm,
            Vθ,
            psid_results_object,
        )

        surr_prob_node_off, P = _initialize_surrogate(   #change naming
            params,
            nn,
            p_nn_init,
            M,
            tsteps,
            fault_dict,
            surr,
            N_ALGEBRAIC_STATES,
            ODE_ORDER,
        )

        if (params.ode_model != "none")
            (params.verify_psid_node_off) && _verify_psid_node_off(
                surr_prob_node_off,
                params,
                solver,
                tsteps,
                fault_dict,
            )
            surr_prob = _turn_node_on(surr_prob_node_off, params, P)
        else
            surr_prob = surr_prob_node_off
        end
        if !(params.learn_initial_condition_unobserved_states)
            x₀_surr = surr_prob.u0
            x₀_surr[(end - params.node_unobserved_states + 1):end] = unobserved_u0
            surr_prob = OrdinaryDiffEq.remake(surr_prob, u0 = x₀_surr)
        end
        fault_data[PSY.get_name(pvs)][:surr_problem] = surr_prob
        fault_data[PSY.get_name(pvs)][:P] = P   #parameters are stored in surr_prob, but as a single vector. The P struct allows for easier handling. 
    end

    θ = partitioned_θ()
    θ.θ_node = p_nn_init #  DiffEqFlux.initial_params(nn)
    θ.θ_observation = observation_params

    if params.learn_initial_condition_unobserved_states
        θ.θ_u0 = rand(
            Int(
                (length(params.training_groups[1][:shoot_times]) + 1) *
                params.node_unobserved_states *
                params.groupsize_faults,
            ),
        )

    else
        θ.θ_u0 = rand(
            Int(
                (length(params.training_groups[1][:shoot_times])) *
                params.node_unobserved_states *
                params.groupsize_faults,
            ),
        )
    end

    @warn "LENGTHS OF θ to start:", combine_θ(θ)[2]

    #try
    total_time = @elapsed begin
        for (fault_group_count, group_pvs) in
            enumerate(IterTools.partition(pvss, params.groupsize_faults))
            pvs_names_subset = PSY.get_name.(group_pvs)
            @info "start of fault group" θ.θ_node[end], pvs_names_subset
            res, output = _train(
                θ,  #partitioned form 
                params,
                sensealg,
                solver,
                solver_tols,
                solver_sensealg,
                optimizer,
                ground_truth_scale,
                output,
                tsteps,
                pvs_names_subset,
                fault_data,
                per_solve_maxiters,
                observation_function,
                fault_group_count,
            )
            θ = res
            @info "end of fault" θ.θ_node[end]
        end

        #TODO - Add second training stage here for final adjustments (optimizer_adjust) 
    end

    output["total_time"] = total_time
    pvs_names = PSY.get_name.(pvss)

    @warn "length of u0 after before", length(θ.θ_u0)
    θ = update_θ_u0(
        θ,
        (
            tspan = (0.0, 0.0),
            shoot_times = [],
            multiple_shoot_continuity_term = (1.0, 1.0),
            batching_sample_factor = 1.0,
        ),
        solver,
        fault_data,
        params,
        PSY.get_name.(pvss),
    )
    @warn "length of u0 after update", length(θ.θ_u0)

    @info "End of training, calculating final loss for comparison:"
    final_loss_for_comparison = _calculate_final_loss(
        params,
        θ,
        solver,
        solver_tols,
        solver_sensealg,
        pvs_names,
        fault_data,
        tsteps,
        ground_truth_scale,
        output,
        observation_function,
    )
    output["final_loss"] = final_loss_for_comparison

    _capture_output(output, params.output_data_path, params.train_id)
    return true
    #TODO - uncomment try catch after debugging 
    #catch
    #    return false
    #end
end

function _train(
    θ::partitioned_θ,
    params::NODETrainParams,
    sensealg::SciMLBase.AbstractADType,
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm,
    solver_tols,
    solver_sensealg,
    optimizer::Union{Flux.Optimise.AbstractOptimiser, Optim.AbstractOptimizer},
    ground_truth_scale::Array{Float64},
    output::Dict{String, Any},
    tsteps::Vector{Float64},
    pvs_names_subset::Tuple,
    fault_data::Dict{String, Dict{Symbol, Any}},
    per_solve_maxiters::Int64,
    observation_function,
    fault_group_count::Int64,
)
    inner_loss_function =
        instantiate_inner_loss_function(params.loss_function_weights, ground_truth_scale)

    res = nothing
    min_θ = θ

    for (training_group_count, training_group) in enumerate(params.training_groups)
        @warn "length of u0 before update", length(min_θ.θ_u0)
        if training_group_count > 1 || fault_group_count > 1
            min_θ = update_θ_u0(
                min_θ,
                training_group,
                solver,
                fault_data,
                params,
                pvs_names_subset,
            )
        end
        @warn "length of u0 after update", length(min_θ.θ_u0)

        θ_vec, θ_lengths = combine_θ(min_θ)
        tspan = training_group[:tspan]
        first_index = findfirst(x -> x >= tspan[1], tsteps)
        last_index = findlast(x -> x <= tspan[2], tsteps)
        range = first_index:last_index
        @info "start of training group" range, min_θ.θ_node[end]

        i_current_range = concatonate_ground_truth(fault_data, pvs_names_subset, range)   #TODO- need to provide all states for Multiple shoot with VSM model? 
        t_current_range = concatonate_t(tsteps, pvs_names_subset, range)

        pvs_ranges = generate_pvs_ranges(pvs_names_subset, length(range))

        batchsize = length(i_current_range[1, :])

        train_loader =
            Flux.Data.DataLoader((i_current_range, t_current_range), batchsize = batchsize)

        outer_loss_function = instantiate_outer_loss_function(
            solver,
            solver_tols,
            solver_sensealg,
            fault_data,
            inner_loss_function,
            training_group,
            θ_lengths,
            params,
            observation_function,
            pvs_names_subset,
            pvs_ranges,
        )
        outer_loss_function(θ_vec, i_current_range, t_current_range)

        optfun = GalacticOptim.OptimizationFunction(
            (θ, p, batch, time_batch) -> outer_loss_function(θ, batch, time_batch),
            sensealg,
        )

        optfun2 = GalacticOptim.instantiate_function(optfun, θ_vec, sensealg, nothing)

        optprob = GalacticOptim.OptimizationProblem(optfun2, θ_vec)

        cb = instantiate_cb!(
            output,
            params.lb_loss,
            params.output_mode,
            params.output_mode_skip,
            training_group_count,
            pvs_names_subset,
        )

        timing_stats_compile = @timed GalacticOptim.solve(
            optprob,
            optimizer,
            IterTools.ncycle(train_loader, 1),
            cb = cb,
        )
        push!(
            output["timing_stats_compile"],
            (
                time = timing_stats_compile.time,
                bytes = timing_stats_compile.bytes,
                gc_time = timing_stats_compile.gctime,
            ),
        )

        timing_stats = @timed GalacticOptim.solve(
            optprob,
            optimizer,
            IterTools.ncycle(train_loader, per_solve_maxiters),
            cb = cb,
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

        min_θ = split_θ(copy(res.u), θ_lengths)
        @info "end of training_group" min_θ.θ_node[end]
    end
    return min_θ, output
end

function generate_pvs_ranges(pvs_names, data_points_per_pvs)
    pvs_ranges = []
    for (i, pvs) in enumerate(pvs_names)
        push!(pvs_ranges, (1 + (i - 1) * data_points_per_pvs):(i * data_points_per_pvs))
    end
    return pvs_ranges
end

function update_θ_u0(
    θ,
    training_group,
    solver,
    fault_data,
    params,
    pvs_names_subset;
    kwargs...,
)
    new_θ_u0 = Float64[]
    saveat = training_group[:shoot_times]

    for pvs in pvs_names_subset
        prob = fault_data[pvs][:surr_problem]
        P = fault_data[pvs][:P]
        P.nn = θ.θ_node
        p = vectorize(P)
        if isempty(saveat)  #FOR FINAL LOSS 
            if params.learn_initial_condition_unobserved_states == false
                xxx = Float64[]
            elseif params.learn_initial_condition_unobserved_states == true
                xxx = θ.θ_u0[1:(params.node_unobserved_states)]
            end
            new_θ_u0 = vcat(new_θ_u0, xxx)
        else #NOT FOR FINAL LOSS 
            if params.learn_initial_condition_unobserved_states == false
                sol = OrdinaryDiffEq.solve(
                    OrdinaryDiffEq.remake(prob; p = p, tspan = (0.0, saveat[end])),
                    solver;
                    saveat = saveat,
                    save_idxs = [
                        length(prob.u0) - params.node_unobserved_states + i for
                        i in 1:(params.node_unobserved_states)
                    ],
                    kwargs...,
                )
            elseif params.learn_initial_condition_unobserved_states == true
                sol = OrdinaryDiffEq.solve(
                    OrdinaryDiffEq.remake(
                        prob;
                        p = p,
                        tspan = (0.0, saveat[end]),
                        u0 = vcat(
                            prob.u0[1:(length(prob.u0) - params.node_unobserved_states)],
                            θ.θ_u0[1:(params.node_unobserved_states)],
                        ),  #Difference: use the initial condition of unobserved state from end of last training.              
                    ),
                    solver;
                    saveat = vcat(0.0, saveat),              #Difference: also save at t = 0
                    save_idxs = [
                        length(prob.u0) - params.node_unobserved_states + i for
                        i in 1:(params.node_unobserved_states)
                    ],
                    kwargs...,
                )
            end
            new_θ_u0 = vcat(new_θ_u0, vec(Array(sol)))
        end
    end
    θ.θ_u0 = new_θ_u0
    return θ
end

#TODO- performance improvements 
function concatonate_ground_truth(fault_data, pvs_names_subset, range)
    ground_truth = []
    for (i, pvs_name) in enumerate(pvs_names_subset)
        ground_truth_fault = fault_data[pvs_name][:ground_truth]
        ground_truth_fault = ground_truth_fault[:, range]
        if i == 1
            ground_truth = ground_truth_fault
        else
            ground_truth = hcat(ground_truth, ground_truth_fault)
        end
    end
    return ground_truth
end

#TODO- performance improvements 
function concatonate_t(tsteps, pvs_names_subset, range)
    t = []
    for (i, pvs_name) in enumerate(pvs_names_subset)
        t_fault = reshape(tsteps[range], 1, length(tsteps[range]))
        if i == 1
            t = t_fault
        else
            t = hcat(t, t_fault)
        end
    end
    return t
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
