using Flux
abstract type GFLLayer <: Function end
Flux.trainable(m::GFLLayer) = (p = m.p,)

struct GFL{PT, PF, PM, SS, DS, A, K} <: GFLLayer
    p_train::PT
    p_fixed::PF
    p_map::PM
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function GFL(  #This is an inner constructor 
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )
        if p === nothing
            p = default_params(PSIDS.GFLParams())
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

Flux.@functor GFL
Flux.trainable(m::GFL) = (p = m.p_train,)

function (s::GFL)(
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

    x0 = zeros(typeof(p_ordered[1]), 15)
    inner_vars = repeat(PSID.ACCEPTED_REAL_TYPES[0.0], 25)
    refs = zeros(typeof(p_ordered[1]), 3)
    ss_solver = s.ss_solver
    ss_tol = s.args[1]
    converged = initialize_dynamic_device!(
        x0,
        inner_vars,
        refs,
        p_ordered,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )

    if converged
        function dudt_dyn!(du, u, p, t)
            device!(du, u, p, refs, inner_vars, V(t)[1], V(t)[2], s)
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
        sol = OrdinaryDiffEq.solve(prob_dyn, s.dyn_solver; s.kwargs...)
        return PhysicalModel_solution(
            tsteps,
            Array(sol[14:15, :]),
            [], #Need residual? 
            true,
        )
    else
        return PhysicalModel_solution(tsteps, [], res, true)
    end
end

function default_params(::PSIDS.GFLParams)
    return Float64[
        690.0,  #rated_voltage
        2.75,   #rated_current
        2.0,    #Kp_p 
        30.0,   #Ki_p
        0.132 * 2 * pi * 50,    #ωz
        2.0,    #Kp_q 
        30.0,   #Ki_q
        0.132 * 2 * pi * 50.0,  #ωf
        0.37,  #kpc
        0.7,    #kic
        1.0,    #kffv
        600.0,  #voltage
        1.32 * 2 * pi * 50, #ω_lp
        2.0,    #kp_pll 
        20.0,   #ki_pll 
        0.009,  #lf 
        0.016,  #rf
        2.50,    #cf
        0.002,  #lg
        0.003,  #rg 
    ]
end

function ordered_param_symbols(::Union{GFL, PSIDS.GFLParams})
    return [
        :rated_voltage_gfl,
        :rated_current_gfl,
        :Kp_p,
        :Ki_p,
        :ωz_gfl,
        :Kp_q,
        :Ki_q,
        :ωf_gfl,
        :kpc_gfl,
        :kic_gfl,
        :kffv_gfl,
        :voltage_gfl,
        :ω_lp,
        :kp_pll,
        :ki_pll,
        :lf_gfl,
        :rf_gfl,
        :cf_gfl,
        :lg_gfl,
        :rg_gfl,
    ]
end

function n_states(::Union{GFL, PSIDS.GFLParams})
    return 15
end

function n_inner_vars(::Union{GFL, PSIDS.GFLParams})
    return 25
end

function n_refs(::Union{GFL, PSIDS.GFLParams})
    return 3
end

function n_params(::Union{GFL, PSIDS.GFLParams})
    return 20
end

function real_current_index(::Union{GFL, PSIDS.GFLParams})
    return 14
end

function imag_current_index(::Union{GFL, PSIDS.GFLParams})
    return 15
end

const gfl_indices = Dict{Symbol, Dict{Symbol, Int64}}(
    :params => Dict{Symbol, Int64}(
        :rated_voltage_gfl => 1,
        :rated_current_gfl => 2,
        :Kp_p => 3,
        :Ki_p => 4,
        :ωz_gfl => 5,
        :Kp_q => 6,
        :Ki_q => 7,
        :ωf_gfl => 8,
        :kpc_gfl => 9,
        :kic_gfl => 10,
        :kffv_gfl => 11,
        :voltage_gfl => 12,
        :ω_lp => 13,
        :kp_pll => 14,
        :ki_pll => 15,
        :lf_gfl => 16,
        :rf_gfl => 17,
        :cf_gfl => 18,
        :lg_gfl => 19,
        :rg_gfl => 20,
    ),
    :states => Dict{Symbol, Int64}(
        :σp_oc => 1,
        :p_oc => 2,
        :σq_oc => 3,
        :q_oc => 4,
        :γd_ic => 5,
        :γq_ic => 6,
        :vq_pll => 7,
        :ε_pll => 8,
        :θ_pll => 9,
        :ir_cnv => 10,
        :ii_cnv => 11,
        :vr_filter => 12,
        :vi_filter => 13,
        :ir_filter => 14,
        :ii_filter => 15,
    ),
    :references => Dict{Symbol, Int64}(:P_ref => 1, :Q_ref => 2, :V_ref => 3),
)

function initialize_dynamic_device!(
    device_states,
    inner_vars,
    references,
    parameters,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    filter_converged = initialize_filter!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )
    frequency_estimator_converged = initialize_frequency_estimator!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )
    outer_converged = initialize_outer!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )
    DCside_converged = initialize_DCside!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )
    converter_converged = initialize_converter!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )
    inner_converged = initialize_inner!(
        device_states,
        inner_vars,
        references,
        parameters,
        v0,
        i0,
        ss_solver,
        ss_tol,
        s,
    )

    return filter_converged &&
           frequency_estimator_converged &&
           outer_converged &&
           DCside_converged &&
           converter_converged &&
           inner_converged
end

function initialize_filter!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    V_R = v0[1]
    V_I = v0[2]
    Ir_filter = i0[1]
    Ii_filter = i0[2]

    #Get Parameters
    #filter = PSY.get_filter(dynamic_device)
    lf = params[gfl_indices[:params][:lf_gfl]]
    rf = params[gfl_indices[:params][:rf_gfl]]
    cf = params[gfl_indices[:params][:cf_gfl]]
    lg = params[gfl_indices[:params][:lg_gfl]]
    rg = params[gfl_indices[:params][:rg_gfl]]

    #Set parameters
    ω_sys = 1.0

    #To solve Vr_cnv, Vi_cnv, Ir_cnv, Ii_cnv, Vr_filter, Vi_filter
    function f!(out, x, params, t)
        Vr_cnv = x[1]
        Vi_cnv = x[2]
        Ir_cnv = x[3]
        Ii_cnv = x[4]
        Vr_filter = x[5]
        Vi_filter = x[6]

        #𝜕Ir_cnv/𝜕t
        out[1] = Vr_cnv - Vr_filter - rf * Ir_cnv + ω_sys * lf * Ii_cnv
        #𝜕Ii_cnv/𝜕t
        out[2] = Vi_cnv - Vi_filter - rf * Ii_cnv - ω_sys * lf * Ir_cnv
        #𝜕Vr_filter/𝜕t
        out[3] = Ir_cnv - Ir_filter + ω_sys * cf * Vi_filter
        #𝜕Vi_filter/𝜕t
        out[4] = Ii_cnv - Ii_filter - ω_sys * cf * Vr_filter
        #𝜕Ir_filter/𝜕t
        out[5] = Vr_filter - V_R - rg * Ir_filter + ω_sys * lg * Ii_filter
        #𝜕Ii_filter/𝜕t
        out[6] = Vi_filter - V_I - rg * Ii_filter - ω_sys * lg * Ir_filter
    end
    x0 = [V_R, V_I, Ir_filter, Ii_filter, V_R, V_I]

    #SOLVE PROBLEM TO STEADY STATE 
    ff_ss = OrdinaryDiffEq.ODEFunction{true}(f!)
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{true}(ff_ss, x0, (zero(x0[1]), one(x0[1]) * 100), params),
    )
    sol = SteadyStateDiffEq.solve(prob_ss, ss_solver; abstol = ss_tol).original

    if !NLsolve.converged(sol)
        @warn("Initialization in Filter failed")
        return false
    else
        sol_x0 = sol.zero
        #Update terminal voltages
        inner_vars[PSID.Vr_inv_var] = V_R
        inner_vars[PSID.Vi_inv_var] = V_I
        #Update Converter voltages
        inner_vars[PSID.Vr_cnv_var] = sol_x0[1]
        inner_vars[PSID.Vi_cnv_var] = sol_x0[2]
        inner_vars[PSID.Ir_cnv_var] = sol_x0[3]
        inner_vars[PSID.Ii_cnv_var] = sol_x0[4]
        #Update filter voltages
        inner_vars[PSID.Vr_filter_var] = sol_x0[5]
        inner_vars[PSID.Vi_filter_var] = sol_x0[6]
        #Update filter currents
        inner_vars[PSID.Ir_filter_var] = Ir_filter
        inner_vars[PSID.Ii_filter_var] = Ii_filter
        #Update states
        device_states[gfl_indices[:states][:ir_cnv]] = sol_x0[3]
        device_states[gfl_indices[:states][:ii_cnv]] = sol_x0[4]
        device_states[gfl_indices[:states][:vr_filter]] = sol_x0[5]
        device_states[gfl_indices[:states][:vi_filter]] = sol_x0[6]
        device_states[gfl_indices[:states][:ir_filter]] = Ir_filter
        device_states[gfl_indices[:states][:ii_filter]] = Ii_filter
    end
    return true
end

function initialize_converter!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    return true
end

function initialize_frequency_estimator!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    Vr_filter = inner_vars[PSID.Vr_filter_var]
    Vi_filter = inner_vars[PSID.Vi_filter_var]

    #Get parameters
    kp_pll = params[gfl_indices[:params][:kp_pll]]
    ki_pll = params[gfl_indices[:params][:ki_pll]]

    #Get initial guess
    θ0_pll = atan(Vi_filter, Vr_filter)
    Vpll_q0 = 0.0
    ϵ_pll0 = 0.0

    function f!(out, x, params, t)
        vpll_q = x[1]
        ϵ_pll = x[2]
        θ_pll = x[3]

        V_dq_pll = PSID.ri_dq(θ_pll + pi / 2) * [Vr_filter; Vi_filter]

        out[1] = V_dq_pll[PSID.q] - vpll_q
        out[2] = vpll_q
        out[3] = kp_pll * vpll_q + ki_pll * ϵ_pll
    end

    x0 = [Vpll_q0, ϵ_pll0, θ0_pll]

    ff_ss = OrdinaryDiffEq.ODEFunction{true}(f!)
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{true}(ff_ss, x0, (zero(x0[1]), one(x0[1]) * 100), params),
    )
    sol = SteadyStateDiffEq.solve(prob_ss, ss_solver; abstol = ss_tol).original

    if !NLsolve.converged(sol)
        @warn("Initialization in PLL failed")
        return false
    else
        sol_x0 = sol.zero

        device_states[gfl_indices[:states][:vq_pll]] = sol_x0[1]
        device_states[gfl_indices[:states][:ε_pll]] = sol_x0[2]
        device_states[gfl_indices[:states][:θ_pll]] = sol_x0[3]

        #Update guess of frequency estimator
        inner_vars[PSID.ω_freq_estimator_var] = 1.0 # get_ω_ref(dynamic_device)
        inner_vars[PSID.θ_freq_estimator_var] = sol_x0[3]
    end
    return true
end

function initialize_outer!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    #Obtain external states inputs for component
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfl_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfl_indices[:states][:ii_filter]]
    Ir_cnv = device_states[gfl_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfl_indices[:states][:ii_cnv]]

    #Obtain additional expressions
    θ0_oc = inner_vars[PSID.θ_freq_estimator_var]
    I_dq_cnv = PSID.ri_dq(θ0_oc + pi / 2) * [Ir_cnv; Ii_cnv]
    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    #Get Outer Controller parameters
    Ki_p = params[gfl_indices[:params][:Ki_p]]
    Ki_q = params[gfl_indices[:params][:Ki_q]]

    #Update inner_vars
    inner_vars[PSID.P_ES_var] = p_elec_out

    #Update states
    device_states[gfl_indices[:states][:σp_oc]] = I_dq_cnv[PSID.q] / Ki_p #σp_oc
    device_states[gfl_indices[:states][:p_oc]] = p_elec_out #p_oc
    device_states[gfl_indices[:states][:σq_oc]] = I_dq_cnv[PSID.d] / Ki_q #σq_oc
    device_states[gfl_indices[:states][:q_oc]] = q_elec_out #q_oc

    #Update inner vars
    inner_vars[PSID.θ_oc_var] = θ0_oc
    inner_vars[PSID.ω_oc_var] = 1.0 #get_ω_ref(dynamic_device)
    inner_vars[PSID.Id_oc_var] = I_dq_cnv[PSID.d]
    inner_vars[PSID.Iq_oc_var] = I_dq_cnv[PSID.q]

    #Update Q_ref. Initialization assumes q_ref = q_elec_out from PF solution
    references[gfl_indices[:references][:P_ref]] = p_elec_out
    references[gfl_indices[:references][:Q_ref]] = q_elec_out
    return true
end

function initialize_DCside!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    inner_vars[PSID.Vdc_var] = params[gfl_indices[:params][:voltage_gfl]]
    return true
end

function initialize_inner!(
    device_states,
    inner_vars,
    references,
    params,
    v0,
    i0,
    ss_solver,
    ss_tol,
    s::Union{GFL, PSIDS.GFLParams},
)
    #Obtain external states inputs for component
    Ir_cnv = device_states[gfl_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfl_indices[:states][:ii_cnv]]
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]

    #Obtain inner variables for component
    ω_oc = 1.0 #get_ω_ref(dynamic_device)
    θ0_oc = inner_vars[PSID.θ_freq_estimator_var]
    Vdc = inner_vars[PSID.Vdc_var]
    Id_cnv_ref = inner_vars[PSID.Id_oc_var]
    Iq_cnv_ref = inner_vars[PSID.Iq_oc_var]

    #Obtain output of converter
    Vr_cnv0 = inner_vars[PSID.Vr_cnv_var]
    Vi_cnv0 = inner_vars[PSID.Vi_cnv_var]

    #Get Current Controller parameters
    kpc = params[gfl_indices[:params][:kpc_gfl]]
    kic = params[gfl_indices[:params][:kic_gfl]]
    kffv = params[gfl_indices[:params][:kffv_gfl]]
    lf = params[gfl_indices[:params][:lf_gfl]]

    function f!(out, x, params, t)
        γ_d = x[1]
        γ_q = x[2]

        #Reference Frame Transformations
        I_dq_cnv = PSID.ri_dq(θ0_oc + pi / 2) * [Ir_cnv; Ii_cnv]
        V_dq_filter = PSID.ri_dq(θ0_oc + pi / 2) * [Vr_filter; Vi_filter]
        V_dq_cnv0 = PSID.ri_dq(θ0_oc + pi / 2) * [Vr_cnv0; Vi_cnv0]

        #References for Converter Output Voltage
        Vd_cnv_ref = (
            kpc * (Id_cnv_ref - I_dq_cnv[PSID.d]) + kic * γ_d -
            ω_oc * lf * I_dq_cnv[PSID.q] + kffv * V_dq_filter[PSID.d]
        )
        Vq_cnv_ref = (
            kpc * (Iq_cnv_ref - I_dq_cnv[PSID.q]) +
            kic * γ_q +
            ω_oc * lf * I_dq_cnv[PSID.d] +
            kffv * V_dq_filter[PSID.q]
        )

        out[1] = Vd_cnv_ref - V_dq_cnv0[PSID.d]
        out[2] = Vq_cnv_ref - V_dq_cnv0[PSID.q]
    end
    x0 = [0.0, 0.0]

    ff_ss = OrdinaryDiffEq.ODEFunction{true}(f!)
    prob_ss = SteadyStateDiffEq.SteadyStateProblem(
        OrdinaryDiffEq.ODEProblem{true}(ff_ss, x0, (zero(x0[1]), one(x0[1]) * 100), params),
    )
    sol = SteadyStateDiffEq.solve(prob_ss, ss_solver; abstol = ss_tol).original

    if !NLsolve.converged(sol)
        @warn("Initialization in Inner Control failed")
        return false
    else
        sol_x0 = sol.zero
        #Update Converter modulation
        m0_dq = (PSID.ri_dq(θ0_oc + pi / 2) * [Vr_cnv0; Vi_cnv0]) ./ Vdc
        inner_vars[PSID.md_var] = m0_dq[PSID.d]
        inner_vars[PSID.mq_var] = m0_dq[PSID.q]

        #Update states
        device_states[gfl_indices[:states][:γd_ic]] = sol_x0[1] #γ_d
        device_states[gfl_indices[:states][:γq_ic]] = sol_x0[2] #γ_q
    end
    return true
end

function device!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    #Obtain global vars
    sys_ω = 1.0 #global_vars[GLOBAL_VAR_SYS_FREQ_INDEX]

    #Update Voltage data
    inner_vars[PSID.Vr_inv_var] = Vr#V(t)[1]
    inner_vars[PSID.Vi_inv_var] = Vi #V(t)[2]

    #Don't use V_ref in GFL equations
    #V_ref = get_V_ref(dynamic_device)
    #inner_vars[PSID.V_oc_var] = V_ref

    #Update current inner_vars
    _update_inner_vars!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    #Obtain ODES for DC side
    mdl_DCside_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    #Obtain ODEs for PLL
    mdl_freq_estimator_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    #Obtain ODEs for OuterLoop
    mdl_outer_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    #Obtain inner controller ODEs and modulation commands
    mdl_inner_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    #Obtain converter relations
    mdl_converter_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    mdl_filter_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFL, PSIDS.GFLParams},
    )

    return
end

function _update_inner_vars!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
) end

function mdl_DCside_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    #Update inner_vars
    inner_vars[PSID.Vdc_var] = p[gfl_indices[:params][:voltage_gfl]]
end

function mdl_freq_estimator_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    ω_sys = 1.0     #MB
    #Obtain external states inputs for component
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]

    #Get parameters
    ω_lp = p[gfl_indices[:params][:ω_lp]]
    kp_pll = p[gfl_indices[:params][:kp_pll]]
    ki_pll = p[gfl_indices[:params][:ki_pll]]
    ωb = 2.0 * pi * 60.0

    #Define internal states for frequency estimator
    vpll_q = device_states[gfl_indices[:states][:vq_pll]]
    ϵ_pll = device_states[gfl_indices[:states][:ε_pll]]
    θ_pll = device_states[gfl_indices[:states][:θ_pll]]

    #Transform to internal dq-PLL reference frame
    V_dq_pll = PSID.ri_dq(θ_pll + pi / 2) * [Vr_filter; Vi_filter]

    #Output Voltage LPF (internal state)
    #𝜕vpll_q/𝜕t, Low Pass Filter, Hug ISGT-EUROPE2018 eqn. 26
    output_ode[gfl_indices[:states][:vq_pll]] =
        PSID.low_pass(V_dq_pll[PSID.q], vpll_q, 1.0, 1.0 / ω_lp)[2]
    #PI Integrator (internal state)
    pi_output, dϵ_dt = PSID.pi_block(vpll_q, ϵ_pll, kp_pll, ki_pll)
    #𝜕dϵ_pll/𝜕t, Hug ISGT-EUROPE2018 eqn. 10
    output_ode[gfl_indices[:states][:ε_pll]] = dϵ_dt
    #PLL Frequency Deviation (internal state), Hug ISGT-EUROPE2018 eqn. 26 
    Δω = 1.0 - ω_sys + pi_output
    #𝜕θ_pll/𝜕t, Hug ISGT-EUROPE2018 eqns. 9, 26, 27 
    output_ode[gfl_indices[:states][:θ_pll]] = ωb * Δω

    #Update inner_vars
    #PLL frequency, D'Arco EPSR122 eqn. 16
    inner_vars[PSID.ω_freq_estimator_var] = Δω + ω_sys
    inner_vars[PSID.θ_freq_estimator_var] = θ_pll
    return
end

function mdl_outer_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfl_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfl_indices[:states][:ii_filter]]

    #Obtain inner variables for component
    θ_pll = inner_vars[PSID.θ_freq_estimator_var]
    ω_pll = inner_vars[PSID.ω_freq_estimator_var]

    #Get Active Power Controller parameters
    Kp_p = p[gfl_indices[:params][:Kp_p]]
    Ki_p = p[gfl_indices[:params][:Ki_p]]
    ωz = p[gfl_indices[:params][:ωz_gfl]]

    #Get Reactive Power Controller parameters
    Kp_q = p[gfl_indices[:params][:Kp_q]]
    Ki_q = p[gfl_indices[:params][:Ki_q]]
    ωf = p[gfl_indices[:params][:ωf_gfl]]

    #Obtain external parameters
    p_ref = references[gfl_indices[:references][:P_ref]]
    q_ref = references[gfl_indices[:references][:Q_ref]]

    #Define internal states for outer control
    σp_oc = device_states[gfl_indices[:states][:σp_oc]]
    p_oc = device_states[gfl_indices[:states][:p_oc]]
    σq_oc = device_states[gfl_indices[:states][:σq_oc]]
    q_oc = device_states[gfl_indices[:states][:q_oc]]

    #Obtain additional expressions
    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    #Compute block derivatives
    Iq_pi, dσpoc_dt = PSID.pi_block(p_ref - p_oc, σp_oc, Kp_p, Ki_p)
    _, dpoc_dt = PSID.low_pass(p_elec_out, p_oc, 1.0, 1.0 / ωz)
    Id_pi, dσqoc_dt = PSID.pi_block(q_ref - q_oc, σq_oc, Kp_q, Ki_q)
    _, dqoc_dt = PSID.low_pass(q_elec_out, q_oc, 1.0, 1.0 / ωf)

    #Compute 4 states ODEs
    output_ode[gfl_indices[:states][:σp_oc]] = dσpoc_dt
    output_ode[gfl_indices[:states][:p_oc]] = dpoc_dt
    output_ode[gfl_indices[:states][:σq_oc]] = dσqoc_dt
    output_ode[gfl_indices[:states][:q_oc]] = dqoc_dt

    #Update inner vars
    inner_vars[PSID.θ_oc_var] = θ_pll
    inner_vars[PSID.ω_oc_var] = ω_pll
    inner_vars[PSID.Iq_oc_var] = Iq_pi
    inner_vars[PSID.Id_oc_var] = Id_pi
    return
end

function mdl_inner_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]
    Ir_cnv = device_states[gfl_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfl_indices[:states][:ii_cnv]]

    #Obtain inner variables for component
    ω_oc = inner_vars[PSID.ω_oc_var]
    θ_oc = inner_vars[PSID.θ_oc_var]
    Vdc = inner_vars[PSID.Vdc_var]

    #Get Current Controller parameters
    kpc = p[gfl_indices[:params][:kpc_gfl]]
    kic = p[gfl_indices[:params][:kic_gfl]]
    kffv = p[gfl_indices[:params][:kffv_gfl]]

    #Get filter params 
    lf = p[gfl_indices[:params][:lf_gfl]]

    γ_d = device_states[gfl_indices[:states][:γd_ic]]
    γ_q = device_states[gfl_indices[:states][:γq_ic]]

    #Transformations to dq frame
    I_dq_cnv = PSID.ri_dq(θ_oc + pi / 2) * [Ir_cnv; Ii_cnv]
    V_dq_filter = PSID.ri_dq(θ_oc + pi / 2) * [Vr_filter; Vi_filter]

    #Input Control Signal - Links to SRF Current Controller
    Id_cnv_ref = inner_vars[PSID.Id_oc_var]
    Iq_cnv_ref = inner_vars[PSID.Iq_oc_var]

    #Current Control PI Blocks
    Vd_pi, dγd_dt = PSID.pi_block(Id_cnv_ref - I_dq_cnv[PSID.d], γ_d, kpc, kic)
    Vq_pi, dγq_dt = PSID.pi_block(Iq_cnv_ref - I_dq_cnv[PSID.q], γ_q, kpc, kic)
    #PI Integrator (internal state)
    output_ode[gfl_indices[:states][:γd_ic]] = dγd_dt
    output_ode[gfl_indices[:states][:γq_ic]] = dγq_dt

    #Compensate references for Converter Output Voltage
    Vd_cnv_ref = Vd_pi - ω_oc * lf * I_dq_cnv[PSID.q] + kffv * V_dq_filter[PSID.d]
    Vq_cnv_ref = Vq_pi + ω_oc * lf * I_dq_cnv[PSID.d] + kffv * V_dq_filter[PSID.q]

    #Update inner_vars
    #Modulation Commands to Converter
    inner_vars[PSID.md_var] = Vd_cnv_ref / Vdc
    inner_vars[PSID.mq_var] = Vq_cnv_ref / Vdc
    return
end

function mdl_converter_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    #Obtain inner variables for component
    md = inner_vars[PSID.md_var]
    mq = inner_vars[PSID.mq_var]
    Vdc = inner_vars[PSID.Vdc_var]
    θ_oc = inner_vars[PSID.θ_oc_var]

    #Transform reference frame to grid reference frame
    m_ri = PSID.dq_ri(θ_oc + pi / 2) * [md; mq]

    #Update inner_vars
    inner_vars[PSID.Vr_cnv_var] = m_ri[PSID.R] * Vdc
    inner_vars[PSID.Vi_cnv_var] = m_ri[PSID.I] * Vdc
    return
end

function mdl_filter_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFL, PSIDS.GFLParams},
)
    #external_ix = get_input_port_ix(dynamic_device, PSY.LCLFilter)
    #θ_oc = device_states[external_ix[1]]

    ω_sys = 1.0 #MB 

    #Obtain inner variables for component
    V_tR = inner_vars[PSID.Vr_inv_var]
    V_tI = inner_vars[PSID.Vi_inv_var]
    Vr_cnv = inner_vars[PSID.Vr_cnv_var]
    Vi_cnv = inner_vars[PSID.Vi_cnv_var]

    #Get parameters
    ωb = 2 * pi * 60.0
    lf = p[gfl_indices[:params][:lf_gfl]]
    rf = p[gfl_indices[:params][:rf_gfl]]
    cf = p[gfl_indices[:params][:cf_gfl]]
    lg = p[gfl_indices[:params][:lg_gfl]]
    rg = p[gfl_indices[:params][:rg_gfl]]

    #Define internal states for filter
    Ir_cnv = device_states[gfl_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfl_indices[:states][:ii_cnv]]
    Vr_filter = device_states[gfl_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfl_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfl_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfl_indices[:states][:ii_filter]]

    #Inputs (control signals) - N/A

    #Compute 6 states ODEs (D'Arco EPSR122 Model)
    #Inverter Output Inductor (internal state)
    #𝜕id_c/𝜕t
    output_ode[gfl_indices[:states][:ir_cnv]] =
        (ωb / lf) * (Vr_cnv - Vr_filter - rf * Ir_cnv + lf * ω_sys * Ii_cnv)
    #𝜕iq_c/𝜕t
    output_ode[gfl_indices[:states][:ii_cnv]] =
        (ωb / lf) * (Vi_cnv - Vi_filter - rf * Ii_cnv - lf * ω_sys * Ir_cnv)
    #LCL Capacitor (internal state)
    #𝜕vd_o/𝜕t
    output_ode[gfl_indices[:states][:vr_filter]] =
        (ωb / cf) * (Ir_cnv - Ir_filter + cf * ω_sys * Vi_filter)
    #𝜕vq_o/𝜕t
    output_ode[gfl_indices[:states][:vi_filter]] =
        (ωb / cf) * (Ii_cnv - Ii_filter - cf * ω_sys * Vr_filter)
    #Grid Inductance (internal state)
    #𝜕id_o/𝜕t
    output_ode[gfl_indices[:states][:ir_filter]] =
        (ωb / lg) * (Vr_filter - V_tR - rg * Ir_filter + lg * ω_sys * Ii_filter)
    #𝜕iq_o/𝜕t
    output_ode[gfl_indices[:states][:ii_filter]] =
        (ωb / lg) * (Vi_filter - V_tI - rg * Ii_filter - lg * ω_sys * Ir_filter)

    #Update inner_vars
    inner_vars[PSID.Vr_filter_var] = Vr_filter
    inner_vars[PSID.Vi_filter_var] = Vi_filter
    return
end
