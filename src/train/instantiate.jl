#TODO use requires to wrap these methods to improve compilation time 
function optimizer_map(key)
    d = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)
    return d[key]
end

function solver_map(key)
    d = Dict("Rodas4" => OrdinaryDiffEq.Rodas4)
    return d[key]
end

function sensealg_map(key)
    d = Dict(
        "ForwardDiff" => GalacticOptim.AutoForwardDiff,
        "Zygote" => GalacticOptim.AutoZygote,
    )   #GalacticOptim.AutoForwardDiff() 
    return d[key]
end

function surr_map(key)
    d = Dict(
        "vsm_2_0_f" => vsm_2_0_f,
        "none_2_0_f" => none_2_0_f,
        "none_2_0_t" => none_2_0_t,
        "none_2_1_f" => none_2_1_f,
        "none_2_2_f" => none_2_2_f,
        "none_2_3_f" => none_2_3_f,
        "none_2_4_f" => none_2_4_f,
        "none_2_5_f" => none_2_5_f,
    )
    return d[key]
end
function activation_map(key)
    d = Dict("relu" => Flux.relu, "hardtanh" => Flux.hardtanh, "sigmoid" => Flux.sigmoid)
    return d[key]
end

function instantiate_solver(inputs)
    return solver_map(inputs.solver)()
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
    nn_input += length(inputs.node_state_inputs)

    nn_output = n_observable_states + inputs.node_unobserved_states
    Random.seed!(inputs.rng_seed)
    @warn "NN size parameters" nn_input, nn_output, nn_width, nn_hidden
    return build_nn(nn_input, nn_output, nn_width, nn_hidden, nn_activation)
end

function build_nn(input_dim, output_dim, nn_width, nn_hidden, nn_activation)
    if nn_hidden == 1
        nn = DiffEqFlux.FastChain(
            DiffEqFlux.FastDense(input_dim, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 2
        nn = DiffEqFlux.FastChain(
            DiffEqFlux.FastDense(input_dim, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 3
        nn = DiffEqFlux.FastChain(
            DiffEqFlux.FastDense(input_dim, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 4
        nn = DiffEqFlux.FastChain(
            DiffEqFlux.FastDense(input_dim, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, output_dim),
        )
        return nn
    elseif nn_hidden == 5
        nn = DiffEqFlux.FastChain(
            DiffEqFlux.FastDense(input_dim, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, nn_width, nn_activation),
            DiffEqFlux.FastDense(nn_width, output_dim),
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

function _instantiate_surr(surr, nn, Vm, Vθ, node_state_inputs)
    return (dx, x, p, t) -> surr(dx, x, p, t, nn, Vm, Vθ, node_state_inputs)
end

function instantiate_node_state_inputs(params, psid_results_object)
    global_indices = Vector{Int64}()
    for p in params.node_state_inputs
        global_index = psid_results_object.global_index[p[1]][p[2]] #Find global index, replace once PSID has a proper interface: https://github.com/NREL-SIIP/PowerSimulationsDynamics.jl/issues/180
        push!(global_indices, global_index)
    end
    return (t) -> (psid_results_object.solution(t, idxs = global_indices))
end

#TODO ALL LOGIC AND NAMING CHANGES 
function instantiate_surr(params, nn, n_observable_states, Vm, Vθ, psid_results_object)
    node_state_inputs = instantiate_node_state_inputs(params, psid_results_object)
    @warn node_state_inputs(0.01)
    number_of_additional_inputs = length(node_state_inputs(0.0))
    @warn number_of_additional_inputs
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
            return _instantiate_surr(surr, nn, Vm, Vθ, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        else
            surr = surr_map(
                string(
                    "vsm_",
                    n_observable_states,
                    "_",
                    params.node_unobserved_states,
                    "_",
                    "f",
                ),
            )
            return _instantiate_surr(surr, nn, Vm, Vθ, node_state_inputs),
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
            return _instantiate_surr(surr, nn, Vm, Vθ, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        else
            surr = surr_map(
                string(
                    "none_",
                    n_observable_states,
                    "_",
                    params.node_unobserved_states,
                    "_",
                    "f",
                ),
            )
            return _instantiate_surr(surr, nn, Vm, Vθ, node_state_inputs),
            N_ALGEBRAIC_STATES,
            ODE_ORDER
        end

    else
        @warn "ode model not found during surrogate instantiatiion"
    end
end

function instantiate_inner_loss_function(loss_function_weights, ground_truth_scale)
    return (u, û) -> _inner_loss_function(u, û, loss_function_weights, ground_truth_scale)
end

function _inner_loss_function(u, û, loss_function_weights, ground_truth_scale)
    n = size(ground_truth_scale, 1)
    loss = 0.0
    for i in 1:n
        loss +=
            mae(û[i, :], u[i, :]) / ground_truth_scale[i] * loss_function_weights[1] +
            mse(û[i, :], u[i, :]) / ground_truth_scale[i] * loss_function_weights[2]
    end
    #=     
        loss =
            (mae(û[1, :], u[1, :]) / Ir_scale) +
            (mae(û[2, :], u[2, :]) / Ii_scale) * loss_function_weights[1] +
            (mse(û[1, :], u[1, :]) / Ir_scale) +
            (mse(û[2, :], u[2, :]) / Ii_scale) * loss_function_weights[2] =#
    return loss
end

function instantiate_outer_loss_function(
    solver,
    fault_data,
    inner_loss_function,
    named_tuple,
)
    return (θ, y_actual, tsteps, pvs_names) -> _outer_loss_function(
        θ,
        y_actual,
        tsteps,
        pvs_names,
        solver,
        fault_data,
        inner_loss_function,
        named_tuple,
    )
end

function _outer_loss_function(
    θ,
    y_actual,
    tsteps,
    pvs_names,
    solver,
    fault_data,
    inner_loss_function,
    named_tuple,
)
    loss = 0.0
    group_predictions = []
    t_predictions = []
    unique_pvs_names = unique(pvs_names)
    #@warn unique_pvs_names
    i = 1
    for pvs in unique_pvs_names
        tsteps_subset = tsteps[pvs .== pvs_names]
        y_actual_subset = y_actual[:, pvs .== pvs_names]
        ms_ranges = shooting_ranges(tsteps, named_tuple[:shoot_times])
        y_actual_subset = eltype(θ).(y_actual_subset)
        tsteps_subset = eltype(θ).(tsteps_subset)      #TODO - can't have this if I want to compare to tsteps with numbers

        P = fault_data[pvs][:P]
        P.nn = θ
        p = vectorize(P)
        single_loss, single_pred, single_t_predictions = batch_multiple_shoot(
            p,
            y_actual_subset,
            tsteps_subset,
            fault_data[pvs][:surr_problem],
            inner_loss_function,
            named_tuple[:multiple_shoot_continuity_term],
            solver,
            ms_ranges,
            named_tuple[:batching_sample_factor],
        )
        loss += single_loss

        if (i == 1)
            group_predictions = single_pred
            t_predictions = single_t_predictions
        else
            group_predictions = vcat(group_predictions, single_pred)
            t_predictions = vcat(t_predictions, single_t_predictions)
        end
        i += 1
    end
    return loss, group_predictions, t_predictions
end

function instantiate_cb!(output, lb_loss, exportmode, range_count, pvs_names) #don't pass t_prediction, let it come from the optimizer? 
    if exportmode == 3
        return (p, l, pred, t) ->
            _cb3!(p, l, pred, t, output, lb_loss, range_count, pvs_names)
    elseif exportmode == 2
        return (p, l, pred, t) -> _cb2!(p, l, output, lb_loss, range_count, pvs_names)
    elseif exportmode == 1
        return (p, l, pred, t) -> _cb1!(p, l, output, lb_loss)
    end
end

function _cb3!(p, l, pred, t_prediction, output, lb_loss, range_count, pvs_names)
    push!(output["loss"], (collect(pvs_names), range_count, l))
    push!(output["parameters"], [p])
    ir = [p[1, :] for p in pred]
    ii = [p[2, :] for p in pred]
    push!(output["predictions"], (t_prediction, ir, ii))
    output["total_iterations"] += 1
    @info "loss", l
    @info "p[end]", p[end]
    (l > lb_loss) && return false
    return true
end

function _cb2!(p, l, output, lb_loss, range_count, pvs_names)
    push!(output["loss"], (collect(pvs_names), range_count, l))
    output["total_iterations"] += 1
    @info "loss", l
    @info "p[end]", p[end]
    (l > lb_loss) && return false
    return true
end

function _cb1!(p, l, output, lb_loss)
    output["total_iterations"] += 1
    @info "loss", l
    @info "p[end]", p[end]
    (l > lb_loss) && return false
    return true
end
