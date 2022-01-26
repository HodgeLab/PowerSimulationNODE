const optimizer_map = Dict("Adam" => Flux.Optimise.ADAM, "Bfgs" => Optim.BFGS)  #use requires - wrap the methods so they are found in the current session 
#These shouldn't be constants, make a function that only tries to find what ADAM is when it iscalled. 
#function that receives the string and returns the method. 

const solver_map = Dict("Rodas4" => OrdinaryDiffEq.Rodas4) #use requires 

const sensealg_map = Dict("ForwardDiffSensitivity" => GalacticOptim.AutoForwardDiff)   #GalacticOptim.AutoForwardDiff() 

const surr_map = Dict(
    "vsm_v_t_0" => vsm_v_t_0,
    "none_v_t_0" => none_v_t_0,
    "none_v_t_1" => none_v_t_1,
    "none_v_t_2" => none_v_t_2,
    "none_v_t_3" => none_v_t_3,
    "none_v_t_4" => none_v_t_4,
    "none_v_t_5" => none_v_t_5,
)

const activation_map =
    Dict("relu" => Flux.relu, "hardtanh" => Flux.hardtanh, "sigmoid" => Flux.sigmoid)

function instantiate_solver(inputs)
    return solver_map[inputs.solver]()
end

function instantiate_sensealg(inputs)
    return sensealg_map[inputs.sensealg]()
end

function instantiate_optimizer(inputs)
    if inputs.optimizer == "Adam"
        return optimizer_map[inputs.optimizer](inputs.optimizer_η)
    elseif inputs.optimizer == "Bfgs"
        return optimizer_map[inputs.optimizer]()
    end
end

function instantiate_optimizer_adjust(inputs)
    if inputs.optimizer_adjust == "Adam"
        return optimizer_map[inputs.optimizer_adjust](inputs.optimizer_adjust_η)
    elseif inputs.optimizer_adjust == "Bfgs"
        return optimizer_map[inputs.optimizer_adjust]()
    end
end

function instantiate_nn(inputs)
    nn_activation = activation_map[inputs.node_activation]
    nn_hidden = inputs.node_layers
    nn_width = inputs.node_width

    nn_input = 4   #P_pf, Q_pf, V_pf, θ_pf
    nn_output = 2

    (inputs.node_inputs == "voltage") && (nn_input += 2)
    (inputs.node_feedback_current) && (nn_input += 2)
    nn_output += inputs.node_feedback_states
    nn_input += inputs.node_feedback_states
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
    n_differential = ODE_ORDER + 2 + inputs.node_feedback_states

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

function _instantiate_surr(surr, nn, Vm, Vθ)
    return (dx, x, p, t) -> surr(dx, x, p, t, nn, Vm, Vθ)
end

function instantiate_surr(inputs, nn, Vm, Vθ)
    if inputs.ode_model == "vsm"
        N_ALGEBRAIC_STATES = 2
        ODE_ORDER = 19
        if inputs.node_inputs == "voltage"
            if inputs.node_feedback_current
                surr = surr_map[string("vsm_v_t_", inputs.node_feedback_states)]
                return _instantiate_surr(surr, nn, Vm, Vθ), N_ALGEBRAIC_STATES, ODE_ORDER
            else
                surr = surr_map[string("vsm_v_f_", inputs.node_feedback_states)]
                return _instantiate_surr(surr, nn, Vm, Vθ), N_ALGEBRAIC_STATES, ODE_ORDER
            end
        else
            @warn "node input type not found during surrogate instantiatiion"
        end
    elseif inputs.ode_model == "none"
        N_ALGEBRAIC_STATES = 0
        ODE_ORDER = 0
        if inputs.node_inputs == "voltage"
            if inputs.node_feedback_current
                surr = surr_map[string("none_v_t_", inputs.node_feedback_states)]
                return _instantiate_surr(surr, nn, Vm, Vθ), N_ALGEBRAIC_STATES, ODE_ORDER
            else
                surr = surr_map[string("none_v_f_", inputs.node_feedback_states)]
                return _instantiate_surr(surr, nn, Vm, Vθ), N_ALGEBRAIC_STATES, ODE_ORDER
            end
        else
            @warn "node input type not found during surrogate instantiatiion"
        end
    else
        @warn "ode model not found during surrogate instantiatiion"
    end
end

function instantiate_inner_loss_function(loss_function_weights, Ir_scale, Ii_scale)
    return (u, û) -> _inner_loss_function(u, û, loss_function_weights, Ir_scale, Ii_scale)
end

function _inner_loss_function(u, û, loss_function_weights, Ir_scale, Ii_scale)
    loss =
        (mae(û[1, :], u[1, :]) / Ir_scale) +
        (mae(û[2, :], u[2, :]) / Ii_scale) * loss_function_weights[1] +
        (mse(û[1, :], u[1, :]) / Ir_scale) +
        (mse(û[2, :], u[2, :]) / Ii_scale) * loss_function_weights[2]
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
    for (i, pvs) in enumerate(unique(pvs_names))
        tsteps_subset = tsteps[pvs .== pvs_names]
        y_actual_subset = y_actual[:, pvs .== pvs_names]
        y_actual_subset = eltype(θ).(y_actual_subset)
        tsteps_subset = eltype(θ).(tsteps_subset)

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
            named_tuple[:multiple_shoot_group_size],
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
    end
    return loss, group_predictions, t_predictions
end

function _loss_function(
    θ,
    y_actual,
    tsteps,
    weights,
    Ir_scale,
    Ii_scale,
    pred_function,
    pvs_names,
)
    y_predicted = pred_function(θ, tsteps, pvs_names)     #Careful of dict ordering?                  

    if size(y_predicted) == size(y_actual)
        loss =
            (mae(y_predicted[1, :], y_actual[1, :]) / Ir_scale) +
            (mae(y_predicted[2, :], y_actual[2, :]) / Ii_scale) * weights[1] +
            (mse(y_predicted[1, :], y_actual[1, :]) / Ir_scale) +
            (mse(y_predicted[2, :], y_actual[2, :]) / Ii_scale) * weights[2]
    else
        loss = Inf
        @warn "Unstable run detected, assigning infinite loss"
    end

    return loss, y_predicted
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
