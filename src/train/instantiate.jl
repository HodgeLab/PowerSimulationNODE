function optimizer_map(key)
    d = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)
    return d[key]
end

function solver_map(key)
    d = Dict(
        "Rodas4" => OrdinaryDiffEq.Rodas4,
        "TRBDF2" => OrdinaryDiffEq.TRBDF2,
        "Tsit5" => OrdinaryDiffEq.Tsit5,
    )
    return d[key]
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

function observation_map(key, n_observable_states)
    d = Dict("first_n" => ((x, θ) -> x[1:n_observable_states, :], Float64[]))
    
    return d[key]
end

function sensealg_map(key)
    d = Dict(
        "ForwardDiff" => GalacticOptim.AutoForwardDiff,
        "Zygote" => GalacticOptim.AutoZygote,
    )
    return d[key]
end

function surr_map(key)
    d = Dict(
        "vsm_2_0_f" => vsm_2_0_f,
        "none_2_0_t" => none_2_0_t,
        "none_2_f_0" => none_2_f_0,
        "none_2_f_0_PQ" => none_2_f_0_PQ,
        "none_2_f_4" => none_2_f_4,
        "none_2_f_4_PQ" => none_2_f_4_PQ,
        "none_2_f_8" => none_2_f_8,
        "none_2_f_8_PQ" => none_2_f_8_PQ,
        "none_2_f_12" => none_2_f_12,
        "none_2_f_12_PQ" => none_2_f_12_PQ,
        "none_2_f_16" => none_2_f_16,
        "none_2_f_16_PQ" => none_2_f_16_PQ,
        "none_3_f_8" => none_3_f_8,
    )
    return d[key]
end

function activation_map(key)
    d = Dict("relu" => Flux.relu, "hardtanh" => Flux.hardtanh, "sigmoid" => Flux.sigmoid)
    return d[key]
end

function instantiate_solver_sensealg(inputs)
    return solver_sensealg_map(inputs.solver_sensealg)
end

function instantiate_solver(inputs)
    return solver_map(inputs.solver)()
end

function instantiate_observation(inputs, n_observable_states)
    return observation_map(inputs.observation_function, n_observable_states)
end

function instantiate_sensealg(inputs)
    return sensealg_map(inputs.sensealg)()
end

function instantiate_optimizer(inputs)
    if inputs.optimizer == "Adam"
        return optimizer_map(inputs.optimizer)(inputs.optimizer_η)
    elseif inputs.optimizer == "Bfgs"
        return optimizer_map(inputs.optimizer)()
    end
end

function instantiate_optimizer_adjust(inputs)
    if inputs.optimizer_adjust == "Adam"
        return optimizer_map(inputs.optimizer_adjust)(inputs.optimizer_adjust_η)
    elseif inputs.optimizer_adjust == "Bfgs"
        return optimizer_map(inputs.optimizer_adjust)()
    end
end

function instantiate_nn(inputs, n_observable_states)
    nn_activation = activation_map(inputs.node_activation)
    nn_hidden = inputs.node_layers
    nn_width = inputs.node_width
    nn_input = n_observable_states + inputs.node_unobserved_states + 6  #P_pf, Q_pf, V_pf, θ_pf, vr(t), vi(t)
    if (inputs.input_PQ)
        nn_input += 2 #P(t), Q(t)
    end
    nn_input += length(inputs.node_state_inputs)

    nn_output = n_observable_states + inputs.node_unobserved_states
    Random.seed!(inputs.rng_seed)
    @warn "NN size parameters" nn_input, nn_output, nn_width, nn_hidden
    return build_nn(nn_input, nn_output, nn_width, nn_hidden, nn_activation)
end

function build_nn(input_dim, output_dim, nn_width, nn_hidden, nn_activation)
    if nn_hidden == 1
        nn = Flux.Chain(
            Flux.Dense(input_dim, nn_width, nn_activation),
            Flux.Dense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 2
        nn = Flux.Chain(
            Flux.Dense(input_dim, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 3
        nn = Flux.Chain(
            Flux.Dense(input_dim, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 4
        nn = Flux.Chain(
            Flux.Dense(input_dim, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 5
        nn = Flux.Chain(
            Flux.Dense(input_dim, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, nn_width, nn_activation),
            Flux.Dense(nn_width, output_dim),
        )
        return nn
    else
        @error "build_nn does not support the provided nn depth"
        return false
    end
end

function instantiate_M(inputs)
    if inputs.ode_model == "vsm"
        ODE_ORDER = 19
        N_ALGEBRAIC = 2
    elseif inputs.ode_model == "none"
        ODE_ORDER = 0
        N_ALGEBRAIC = 0
    else
        @error "ODE order unknown for ODE model provided"
    end
    n_differential = ODE_ORDER + 2 + inputs.node_unobserved_states

    return MassMatrix(n_differential, N_ALGEBRAIC)
end

"""

Makes a Float64 Mass Matrix of ones for the ODEProblem. Takes # of differential and algebraic states.
The algebraic states come first.

"""
function MassMatrix(n_differential::Integer, n_algebraic::Integer)
    n_states = n_differential + n_algebraic
    M = Float64.(zeros(n_states, n_states))
    for i in (n_algebraic + 1):(n_differential + n_algebraic)
        M[i, i] = 1.0
    end
    return M
end

function _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs)
    return (dx, x, p, t) -> surr(dx, x, p, t, nn, Vm, Vθ, n_params_nn, node_state_inputs)
end

function instantiate_node_state_inputs(params, psid_results_object)
    global_indices = Vector{Int64}()
    for p in params.node_state_inputs
        global_index = psid_results_object.global_index[p[1]][p[2]] #TODO: Find global index, replace once PSID has a proper interface: https://github.com/NREL-SIIP/PowerSimulationsDynamics.jl/issues/180
        push!(global_indices, global_index)
    end
    return (t) -> (psid_results_object.solution(t, idxs = global_indices))
end

function instantiate_surr(
    params,
    nn,
    nn_params,
    n_observable_states,
    Vm,
    Vθ,
    psid_results_object,
)
    node_state_inputs = instantiate_node_state_inputs(params, psid_results_object)
    @info "Number of additional state inputs:", length(node_state_inputs(0.01))
    number_of_additional_inputs = length(node_state_inputs(0.0))
    n_params_nn = length(nn_params)
    if params.ode_model == "vsm"
        N_ALGEBRAIC_STATES = 2
        ODE_ORDER = 19
        if number_of_additional_inputs > 0
            surr = surr_map(
                string(
                    "vsm_",
                    n_observable_states,
                    "_",
                    params.node_unobserved_states,
                    "_",
                    "t",
                ),
            )
            return _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        else
            surr = surr_map(string("vsm_", n_observable_states, "_", "f"))
            return _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        end
    elseif params.ode_model == "none"
        N_ALGEBRAIC_STATES = 0
        ODE_ORDER = 0
        if number_of_additional_inputs > 0
            surr = surr_map(
                string(
                    "none_",
                    n_observable_states,
                    "_",
                    params.node_unobserved_states,
                    "_",
                    "t",
                ),
            )
            return _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        else
            if params.input_PQ
                surr = surr_map(
                    string(
                        "none_",
                        n_observable_states,
                        "_",
                        "f_",
                        params.node_unobserved_states,
                        "_PQ",
                    ),
                )
                return _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs),
                N_ALGEBRAIC_STATES,
                ODE_ORDER
            else
                surr = surr_map(
                    string(
                        "none_",
                        n_observable_states,
                        "_",
                        "f_",
                        params.node_unobserved_states,
                    ),
                )
                return _instantiate_surr(surr, nn, Vm, Vθ, n_params_nn, node_state_inputs),
                N_ALGEBRAIC_STATES,
                ODE_ORDER
            end
        end

    else
        @warn "ode model not found during surrogate instantiatiion"
    end
end

function instantiate_inner_loss_function(loss_function_weights, ground_truth_scale)
    return (u, û) -> _inner_loss_function(u, û, loss_function_weights, ground_truth_scale)
end

function _inner_loss_function(u, û, loss_function_weights, ground_truth_scale)
    n_obs = size(ground_truth_scale, 1)
    n_preds = size(u, 1)
    loss = 0.0
    for i in 1:n_obs
        loss +=
            sum(abs, û[i, :] .- u[i, :]) / ground_truth_scale[i] *
            loss_function_weights[1] +
            sum(abs2, û[i, :] .- u[i, :]) / ground_truth_scale[i] *
            loss_function_weights[2]
    end
    for i in (n_obs + 1):n_preds
        loss +=
            sum(abs, û[i, :] .- u[i, :]) * loss_function_weights[1] +
            sum(abs2, û[i, :] .- u[i, :]) * loss_function_weights[2]
    end
    return loss
end

function instantiate_outer_loss_function(
    solver,
    solver_tols,
    solver_sensealg,
    fault_data,
    inner_loss_function,
    training_group,
    θ_lengths,
    params,
    observation_function,
    pvs_names,
    pvs_ranges,
)
    return (θ, y_actual, tsteps) -> _outer_loss_function(
        θ,
        y_actual,
        tsteps,
        solver,
        solver_tols,
        solver_sensealg,
        fault_data,
        inner_loss_function,
        training_group,
        θ_lengths,
        params,
        observation_function,
        pvs_names,
        pvs_ranges,
    )
end

function _outer_loss_function(
    θ_vec,
    y_actual,
    tsteps,
    solver,
    solver_tols,
    solver_sensealg,
    fault_data,
    inner_loss_function,
    training_group,
    θ_lengths,
    params,
    observation_function,
    pvs_names,
    pvs_ranges,
)
    loss = 0.0
    group_predictions = []
    group_observations = []
    t_predictions = []

    for (i, pvs_name) in enumerate(pvs_names)
        tsteps_subset = tsteps[pvs_ranges[i]]
        y_actual_subset = y_actual[:, pvs_ranges[i]]
        ms_ranges = shooting_ranges(tsteps, training_group[:shoot_times])   #includes the starting range (t=0)
        θ = split_θ(θ_vec, θ_lengths)
        θ_u0_subset = split_θ_u0(θ.θ_u0, i, length(pvs_names)) #length(pvs_range_dict)) #

        single_loss, single_pred, single_observation, single_t_predictions =
            batch_multiple_shoot(
                θ.θ_node,
                θ_u0_subset,
                θ.θ_observation,
                y_actual_subset,
                tsteps_subset,
                fault_data[pvs_name],
                inner_loss_function,
                training_group[:multiple_shoot_continuity_term],
                solver,
                solver_tols,
                solver_sensealg,
                ms_ranges,
                training_group[:batching_sample_factor],
                params,
                observation_function,
            )
        loss += single_loss

        if (i == 1)
            group_predictions = single_pred
            group_observations = single_observation
            t_predictions = single_t_predictions
        else
            group_predictions = vcat(group_predictions, single_pred)
            group_observations = vcat(group_observations, single_observation)
            t_predictions = vcat(t_predictions, single_t_predictions)
        end
    end

    return loss, group_predictions, group_observations, t_predictions
end

function instantiate_cb!(
    output,
    lb_loss,
    exportmode,
    exportmode_skip,
    range_count,
    pvs_names,
)
    if Sys.iswindows() || Sys.isapple()
        print_loss = true
    else
        print_loss = false
    end

    if exportmode == 3
        return (p, l, pred, obs, t) -> _cb3!(
            p,
            l,
            pred,
            obs,
            t,
            output,
            lb_loss,
            range_count,
            pvs_names,
            print_loss,
            exportmode_skip,
        )
    elseif exportmode == 2
        return (p, l, pred, obs, t) ->
            _cb2!(p, l, output, lb_loss, range_count, pvs_names, print_loss)
    elseif exportmode == 1
        return (p, l, pred, obs, t) -> _cb1!(p, l, output, lb_loss, print_loss)
    end
end

function _cb3!(
    p,
    l,
    pred,
    obs,
    t_prediction,
    output,
    lb_loss,
    range_count,
    pvs_names,
    print_loss,
    exportmode_skip,
)
    push!(output["loss"], (collect(pvs_names), range_count, l))
    output["total_iterations"] += 1
    if mod(output["total_iterations"], exportmode_skip) == 0
        push!(output["parameters"], [p])
        push!(output["predictions"], (t_prediction, pred, obs))
        push!(output["recorded_iterations"], output["total_iterations"]) #Indices of the recorded iterations... 
    end

    if (print_loss)
        println(l)
    end
    (l > lb_loss) && return false
    return true
end

function _cb2!(p, l, output, lb_loss, range_count, pvs_names, print_loss)
    push!(output["loss"], (collect(pvs_names), range_count, l))
    output["total_iterations"] += 1
    if (print_loss)
        println(l)
    end
    (l > lb_loss) && return false
    return true
end

function _cb1!(p, l, output, lb_loss, print_loss)
    output["total_iterations"] += 1
    if (print_loss)
        println(l)
    end
    (l > lb_loss) && return false
    return true
end
