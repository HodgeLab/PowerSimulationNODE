using Flux
abstract type ClassicGenLayer <: Function end
Flux.trainable(m::ClassicGenLayer) = (p = m.p,)

struct ClassicGen{PT, PF, PM, SS, DS, A, K} <: ClassicGenLayer
    p_train::PT
    p_fixed::PF
    p_map::PM
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function ClassicGen(  #This is an inner constructor 
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )
        if p === nothing
            p = Float64[0.0, 0.2995, 0.7087, 3.148, 2.0] #default starting parameters 
        end
        new{
            typeof(p),
            typeof(p),
            Vector{Int64},
            typeof(ss_solver),
            typeof(dyn_solver),
            typeof(args),
            typeof(kwargs),
        }(
            p,
            [],
            1:length(p),
            ss_solver,
            dyn_solver,
            args,
            kwargs,
        )
    end
end

Flux.@functor ClassicGen
Flux.trainable(m::ClassicGen) = (p = m.p_train,)

function (s::ClassicGen)(
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

    P0 = v0[1] * i0[1] + v0[2] * i0[2]
    Q0 = v0[2] * i0[1] - v0[1] * i0[2]

    R, Xd_p, eq_p, H, D = p_ordered
    V_complex = v0[1] + v0[2] * 1im
    I_complex = i0[1] + i0[2] * 1im
    δ0 = angle(V_complex + (R + Xd_p * 1im) * I_complex)
    ω0 = 1.0
    τm0 = real(V_complex * conj(I_complex))
    u0_pred = [δ0, τm0, 1.0]

    #TODO - organize like the GFL, the classical gen was just a proof of concept 
    #TODO - eq_p shouldn't be a parameter  
    function dudt_ss(u, p, t)
        R, Xd_p, eq_p, H, D = p
        δ, τm, Vf0 = u

        ri_dq = [sin(δ) -cos(δ); cos(δ) sin(δ)]
        V_dq = ri_dq * V(0.0)
        i_d = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * (Vf0 - V_dq[2]) - R * V_dq[1])  #15.36 # (1.0 / (R^2 + Xd_p^2)) *
        i_q = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * V_dq[1] + R * (Vf0 - V_dq[2])) #15.36 #(1.0 / (R^2 + Xd_p^2)) * 
        Pe = (V_dq[1] + R * i_d) * i_d + (V_dq[2] + R * i_q) * i_q
        return vcat(
            τm - Pe, #Mechanical Torque
            P0 - (V_dq[1] * i_d + V_dq[2] * i_q), #Output Power
            Q0 - (V_dq[2] * i_d - V_dq[1] * i_q), #Output Reactive Power
        )
    end

    #This is the dynamics i.e. dx/dt = f(x)
    function dudt_dyn(u, p, t)
        R, Xd_p, _, H, D = p
        δ, ω = u

        τm0 = refs[1]
        eq_p = refs[2]
        ri_dq = [sin(δ) -cos(δ); cos(δ) sin(δ)]
        f0 = 60
        ω_sys = 1.0 #TODO should ω be an additional input to surrogates? 
        V_dq = ri_dq * V(t)
        i_d = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * (eq_p - V_dq[2]) - R * V_dq[1])  #15.36
        i_q = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * V_dq[1] + R * (eq_p - V_dq[2])) #15.36
        Pe = (V_dq[1] + R * i_d) * i_d + (V_dq[2] + R * i_q) * i_q      #15.35
        τe = Pe
        return vcat(
            2 * π * f0 * (ω - ω_sys),                    #15.5
            (1 / (2 * H)) * (τm0 - τe - D * (ω - 1.0) / ω),
        )
    end

    #SOLVE PROBLEM TO STEADY STATE 
    ff_ss = OrdinaryDiffEq.ODEFunction{false}(dudt_ss)
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{false}(
            ff_ss,
            u0_pred,
            (zero(u0_pred[1]), one(u0_pred[1]) * 100),
            p_ordered,
        ),
    )
    ss_solution = SteadyStateDiffEq.solve(prob_ss, s.ss_solver; abstol = s.args[1])
    #=     display(s.args[1])
        display(ss_solution.original)
        display(dudt_ss(u0_pred, p, 0.0))
        display(ss_solution.u) =#

    res = dudt_ss(u0_pred, p, 0.0)

    #TODO - extra call (dummy) to propogate gradients needed after ss_solution is reached? Not sure if needed. 
    #https://github.com/SciML/DeepEquilibriumNetworks.jl/blob/9c2626d6080bbda3c06b81d2463744f5e395003f/src/layers/deq.jl#L41
    #@error NLsolve.converged(ss_solution.original) #check if SS condition was foudn. 
    if NLsolve.converged(ss_solution.original)
        #SOLVE DYNAMICS
        refs = ss_solution.u[(end - 1):end] #NOTE: refs is used in dudt_dyn
        ff = OrdinaryDiffEq.ODEFunction{false}(dudt_dyn) #,tgrad=basic_tgrad)    
        prob_dyn = OrdinaryDiffEq.ODEProblem{false}(
            ff,
            [ss_solution.u[1], 1.0], #δ, ω
            (tsteps[1], tsteps[end]),
            p_ordered;
            tstops = tstops,
        )

        function f_saving(u, t, integrator)
            R, Xd_p, _, _, _ = p
            δ = u[1]
            eq_p = refs[2]
            ri_dq = [sin(δ) -cos(δ); cos(δ) sin(δ)]
            V_dq = ri_dq * V(t)
            i_d = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * (eq_p - V_dq[2]) - R * V_dq[1])  #15.36
            i_q = (1.0 / (R^2 + Xd_p^2)) * (Xd_p * V_dq[1] + R * (eq_p - V_dq[2])) #15.36
            ir, ii = [sin(δ) cos(δ); -cos(δ) sin(δ)] * [i_d; i_q]
            return (ir, ii)
        end

        saved_values = DiffEqCallbacks.SavedValues(
            typeof(tsteps[1]),
            Tuple{typeof(ss_solution.u[1]), typeof(ss_solution.u[1])},
        )
        cb = DiffEqCallbacks.SavingCallback(f_saving, saved_values; saveat = tsteps)
        sol = OrdinaryDiffEq.solve(prob_dyn, s.dyn_solver, callback = cb; s.kwargs...)
        return PhysicalModel_solution(
            tsteps,
            vcat(
                [s[1] for s in saved_values.saveval]',
                [s[2] for s in saved_values.saveval]',
            ),
            res,
            true,
        )

    else
        return PhysicalModel_solution(tsteps, [], [1.0], true)
    end
end
