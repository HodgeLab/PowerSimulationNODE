using Flux
abstract type MultiDeviceLayer <: Function end
Flux.trainable(m::MultiDeviceLayer) = (p = m.p,)

struct MultiDevice{PT, PF, PM, SD, DD, SS, DS, A, K} <: MultiDeviceLayer
    p_train::PT
    p_fixed::PF
    p_map::PM
    static_devices::SD
    dynamic_devices::DD
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function MultiDevice(  #This is an inner constructor 
        static_devices,
        dynamic_devices,
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )
        if p === nothing
            p = Float64[]
            #p_static, q_static, p_gfl, q_gfl, p_gfm, q_gfm
            p = vcat(p, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) #ones(2 * (length(static_devices) + length(dynamic_devices))))    #even distribution of P,Q across devices.
            for d in static_devices
                p = vcat(p, default_params(d))
            end
            for d in dynamic_devices
                p = vcat(p, default_params(d))
            end
        elseif length(p) == ones(2 * (length(static_devices) + length(dynamic_devices)))
            for d in static_devices
                p = vcat(p, default_params(d))
            end
            for d in dynamic_devices
                p = vcat(p, default_params(d))
            end
        end
        new{
            typeof(p),
            typeof(p),
            Vector{Int64},
            Vector{PSIDS.SurrogateModelParams},
            Vector{PSIDS.SurrogateModelParams},
            typeof(ss_solver),
            typeof(dyn_solver),
            typeof(args),
            typeof(kwargs),
        }(
            p,
            [],
            1:length(p),
            static_devices,
            dynamic_devices,
            ss_solver,
            dyn_solver,
            args,
            kwargs,
        )
    end
end

Flux.@functor MultiDevice
Flux.trainable(m::MultiDevice) = (p = m.p_train,)

function (s::MultiDevice)(
    V,
    v0,
    i0,
    tsteps,
    tstops,
    p_fixed = s.p_fixed,
    p_train = s.p_train,
    p_map = s.p_map,
)
    p = vcat(p_fixed, p_train)
    p_ordered = p[p_map]
    ss_tol = s.args[1]
    ss_solver = s.ss_solver

    x0 = zeros(typeof(p_ordered[1]), sum([n_states(x) for x in s.dynamic_devices]))
    inner_vars = repeat(
        PSID.ACCEPTED_REAL_TYPES[0.0],
        sum([n_inner_vars(x) for x in s.dynamic_devices]),
    )
    refs = zeros(
        typeof(p_ordered[1]),
        sum([n_refs(x) for x in vcat(s.static_devices, s.dynamic_devices)]),
    )

    converged = true
    inner_var_start_index = 1
    ref_start_index = 1
    state_start_index = 1
    p_start_index = 1
    #PARAMETERS (FIXED) FOR DISTRIBUTING P0,Q0 amongst devices
    p_end_index =
        p_start_index + 2 * (length(s.static_devices) + length(s.dynamic_devices)) - 1

    i0_static, i0_dynamic = calculate_distributed_i0(
        p_ordered[p_start_index:p_end_index],
        v0,
        i0,
        s.static_devices,
        s.dynamic_devices,
    )
    p_start_index = p_end_index + 1

    #INITIALIZE STATIC DEVICES 
    for (ix, s) in enumerate(s.static_devices)
        ref_end_index = ref_start_index + n_refs(s) - 1
        p_end_index = p_start_index + n_params(s) - 1
        initilize_static_device!(
            view(refs, ref_start_index:ref_end_index),
            view(p_ordered, p_start_index:p_end_index),
            v0,
            i0_static[ix],
        )
        ref_start_index = ref_end_index + 1
        p_start_index = p_end_index + 1
    end

    #INITIALIZE DYNAMIC DEVICES 
    for (ix, s) in enumerate(s.dynamic_devices)
        state_end_index = state_start_index + n_states(s) - 1
        inner_var_end_index = inner_var_start_index + n_inner_vars(s) - 1
        ref_end_index = ref_start_index + n_refs(s) - 1
        p_end_index = p_start_index + n_params(s) - 1
        converged = initialize_dynamic_device!(
            view(x0, state_start_index:state_end_index),
            view(inner_vars, inner_var_start_index:inner_var_end_index),
            view(refs, ref_start_index:ref_end_index),
            view(p_ordered, p_start_index:p_end_index),
            v0,
            i0_dynamic[ix],
            ss_solver,
            ss_tol,
            s,
        )
        state_start_index = state_end_index + 1
        inner_var_start_index = inner_var_end_index + 1
        ref_start_index = ref_end_index + 1
        p_start_index = p_end_index + 1
        (!converged && break)
    end

    if converged
        function dudt_dyn!(du, u, p, t)
            Vr = V(t)[1]
            Vi = V(t)[2]
            inner_var_start_index = 1
            ref_start_index = 1
            state_start_index = 1
            p_start_index = 7  # 6 power splitting parameters
            for s in s.static_devices
                ref_end_index = ref_start_index + n_refs(s) - 1
                p_end_index = p_start_index + n_params(s) - 1

                ref_start_index = ref_end_index + 1
                p_start_index = p_end_index + 1
            end

            for s in s.dynamic_devices
                state_end_index = state_start_index + n_states(s) - 1
                inner_var_end_index = inner_var_start_index + n_inner_vars(s) - 1
                ref_end_index = ref_start_index + n_refs(s) - 1
                p_end_index = p_start_index + n_params(s) - 1
                device!(
                    view(du, state_start_index:state_end_index),
                    view(u, state_start_index:state_end_index),
                    view(p, p_start_index:p_end_index),
                    view(refs, ref_start_index:ref_end_index),
                    view(inner_vars, inner_var_start_index:inner_var_end_index),
                    Vr,
                    Vi,
                    s,
                )
                state_start_index = state_end_index + 1
                inner_var_start_index = inner_var_end_index + 1
                ref_start_index = ref_end_index + 1
                p_start_index = p_end_index + 1
            end
        end
        du = similar(x0)
        dudt_dyn!(du, x0, p_ordered, 0.0)    #DEBUG - this is the residual

        function f_saving(u, t, integrator)
            Vr = V(t)[1]
            Vi = V(t)[2]
            ir_device_total = 0.0
            ii_device_total = 0.0
            ref_start_index = 1
            state_start_index = 1
            p_start_index = 7   #6 power splitting parameters
            for s in s.static_devices
                ref_end_index = ref_start_index + n_refs(s) - 1
                p_end_index = p_start_index + n_params(s) - 1
                ir_device_total -= device(
                    view(p, p_start_index:p_end_index),
                    view(refs, ref_start_index:ref_end_index),
                    Vr,
                    Vi,
                    s,
                )[1]    #POLARITY OF LOAD 
                ii_device_total -= device(
                    view(p, p_start_index:p_end_index),
                    view(refs, ref_start_index:ref_end_index),
                    Vr,
                    Vi,
                    s,
                )[2]
                ref_start_index = ref_end_index + 1
                p_start_index = p_end_index + 1
            end
            for s in s.dynamic_devices
                state_end_index = state_start_index + n_states(s) - 1
                ir_device_total +=
                    view(u, state_start_index:state_end_index)[real_current_index(s)]
                ii_device_total +=
                    view(u, state_start_index:state_end_index)[imag_current_index(s)]
                state_start_index = state_end_index + 1
            end
            return (ir_device_total, ii_device_total)
        end

        ff = OrdinaryDiffEq.ODEFunction{true}(dudt_dyn!; tgrad = basic_tgrad_inplace)
        prob_dyn = OrdinaryDiffEq.ODEProblem{true}(
            ff,
            x0,
            eltype(p_ordered).((tsteps[1], tsteps[end])),
            p_ordered;
            tstops = tstops,
            saveat = tsteps,
        )
        saved_values = DiffEqCallbacks.SavedValues(Any, Tuple{Any, Any})
        cb = DiffEqCallbacks.SavingCallback(f_saving, saved_values; saveat = tsteps)
        sol = OrdinaryDiffEq.solve(prob_dyn, s.dyn_solver, callback = cb; s.kwargs...)
        return PhysicalModel_solution(
            tsteps,
            vcat(
                [s[1] for s in saved_values.saveval]',
                [s[2] for s in saved_values.saveval]',
            ),
            [1.0],
            true,
        )

    else
        return PhysicalModel_solution(tsteps, [], [1.0], true)
    end
end

#Helper function to go between VI/PQ. Need PQ to distribute powers among devices, and VI is how the devices are modeled. 
function PQV_to_I(P, Q, V)
    ir = (P * V[1] + V[2] * Q) / (V[1]^2 + V[2]^2)
    ii = (P * V[2] - V[1] * Q) / (V[1]^2 + V[2]^2)
    return [ir, ii]
end

function VI_to_PQ(V, I)
    P = real((V[1] + im * V[2]) * conj((I[1] + im * I[2])))
    Q = imag((V[1] + im * V[2]) * conj((I[1] + im * I[2])))
    return P, Q
end

#Takes V0, I0 for the entire device, and splits up current according to device based on parameter for proportion of P and Q. 
function calculate_distributed_i0(p, v0, i0, static_devices, dynamic_devices)
    param_index = 1
    p_frac_devices = p[1:2:end]
    q_frac_devices = p[2:2:end]
    i0_static = []
    i0_dynamic = []
    P0, Q0 = VI_to_PQ(v0, i0)
    for s in static_devices
        P0_device = p_frac_devices[param_index] / sum(p_frac_devices) * P0
        Q0_device = q_frac_devices[param_index] / sum(q_frac_devices) * Q0
        push!(i0_static, PQV_to_I(P0_device, Q0_device, v0))
        param_index += 1
    end
    for s in dynamic_devices
        P0_device = p_frac_devices[param_index] / sum(p_frac_devices) * P0
        Q0_device = q_frac_devices[param_index] / sum(q_frac_devices) * Q0
        push!(i0_dynamic, PQV_to_I(P0_device, Q0_device, v0))
        param_index += 1
    end
    return i0_static, i0_dynamic
end
