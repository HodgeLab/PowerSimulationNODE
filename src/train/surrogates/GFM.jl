using Flux
abstract type GFMLayer <: Function end
Flux.trainable(m::GFMLayer) = (p = m.p,)

struct GFM{PT, PF, PM, SS, DS, A, K} <: GFMLayer
    p_train::PT
    p_fixed::PF
    p_map::PM
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function GFM(  #This is an inner constructor 
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )
        if p === nothing
            p = default_params(PSIDS.GFMParams())
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

Flux.@functor GFM
Flux.trainable(m::GFM) = (p = m.p_train,)

function (s::GFM)(
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

    x0 = zeros(typeof(p_ordered[1]), n_states(s))
    inner_vars = repeat(PSID.ACCEPTED_REAL_TYPES[0.0], n_inner_vars(s))
    refs = zeros(typeof(p_ordered[1]), n_refs(s))
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
            Array(sol[real_current_index(s):imag_current_index(s), :]),
            [],
            true,
        )
    else
        return PhysicalModel_solution(tsteps, [1.0], res, true)
    end
end

function default_params(::PSIDS.GFMParams)
    return Float64[
        690.0,  #rated_voltage
        2.75,   #rated_current
        0.05,    #Rp 
        2 * pi * 5,   #ωz
        2.0,    #Kq 
        1000.0,  #ωf
        0.59,   #kpv
        736.0,   #kiv 
        0.0,    #kffv (fix)
        0.0,   #rv
        0.2,    #lv
        1.27,   #kpc
        14.3,   #kic
        0.0,    #kffi
        50.0,   #ωad
        0.2,    #kad
        600.0,  #voltage
        0.08,  #lf 
        0.003,  #rf
        0.074,    #cf
        0.2,  #lg
        0.01,  #rg 
    ]
end

function ordered_param_symbols(::Union{GFM, PSIDS.GFMParams})
    return [
        :rated_voltage_gfm,
        :rated_current_gfm,
        :Rp,
        :ωz_gfm,
        :kq,
        :ωf_gfm,
        :kpv,
        :kiv,
        :kffv_gfm,
        :rv,
        :lv,
        :kpc_gfm,
        :kic_gfm,
        :kffi,
        :ωad,
        :kad,
        :voltage_gfm,
        :lf_gfm,
        :rf_gfm,
        :cf_gfm,
        :lg_gfm,
        :rg_gfm,
    ]
end

function n_states(::Union{GFM, PSIDS.GFMParams})
    return 15
end

function n_inner_vars(::Union{GFM, PSIDS.GFMParams})
    return 25
end

function n_refs(::Union{GFM, PSIDS.GFMParams})
    return 3
end

function n_params(::Union{GFM, PSIDS.GFMParams})
    return 22
end

function real_current_index(::Union{GFM, PSIDS.GFMParams})
    return 14
end

function imag_current_index(::Union{GFM, PSIDS.GFMParams})
    return 15
end

const gfm_indices = Dict{Symbol, Dict{Symbol, Int64}}(
    :params => Dict{Symbol, Int64}(
        :rated_voltage_gfm => 1,
        :rated_current_gfm => 2,
        :Rp => 3,
        :ωz_gfm => 4,
        :kq => 5,
        :ωf_gfm => 6,
        :kpv => 7,
        :kiv => 8,
        :kffv_gfm => 9,
        :rv => 10,
        :lv => 11,
        :kpc_gfm => 12,
        :kic_gfm => 13,
        :kffi => 14,
        :ωad => 15,
        :kad => 16,
        :voltage_gfm => 17,
        :lf_gfm => 18,
        :rf_gfm => 19,
        :cf_gfm => 20,
        :lg_gfm => 21,
        :rg_gfm => 22,
    ),
    :states => Dict{Symbol, Int64}(
        :θ_oc => 1,
        :p_oc => 2,
        :q_oc => 3,
        :ξd_ic => 4,
        :ξq_ic => 5,
        :γd_ic => 6,
        :γq_ic => 7,
        :ϕd_ic => 8,
        :ϕq_ic => 9,
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
    s::Union{GFM, PSIDS.GFMParams},
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
    s::Union{GFM, PSIDS.GFMParams},
)
    V_R = v0[1]
    V_I = v0[2]
    Ir_filter = i0[1]
    Ii_filter = i0[2]

    #Get Parameters
    #filter = PSY.get_filter(dynamic_device)
    lf = params[gfm_indices[:params][:lf_gfm]]
    rf = params[gfm_indices[:params][:rf_gfm]]
    cf = params[gfm_indices[:params][:cf_gfm]]
    lg = params[gfm_indices[:params][:lg_gfm]]
    rg = params[gfm_indices[:params][:rg_gfm]]

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
        false
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
        device_states[gfm_indices[:states][:ir_cnv]] = sol_x0[3]
        device_states[gfm_indices[:states][:ii_cnv]] = sol_x0[4]
        device_states[gfm_indices[:states][:vr_filter]] = sol_x0[5]
        device_states[gfm_indices[:states][:vi_filter]] = sol_x0[6]
        device_states[gfm_indices[:states][:ir_filter]] = Ir_filter
        device_states[gfm_indices[:states][:ii_filter]] = Ii_filter
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
    s::Union{GFM, PSIDS.GFMParams},
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
    s::Union{GFM, PSIDS.GFMParams},
)
    #Update guess of frequency estimator
    inner_vars[PSID.ω_freq_estimator_var] = 1.0
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
    s::Union{GFM, PSIDS.GFMParams},
)
    #Obtain external states inputs for component
    #Obtain external states inputs for component
    Vr_filter = device_states[gfm_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfm_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfm_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfm_indices[:states][:ii_filter]]

    Vr_cnv = inner_vars[PSID.Vr_cnv_var]
    Vi_cnv = inner_vars[PSID.Vi_cnv_var]
    θ0_oc = atan(Vi_cnv, Vr_cnv)

    #Obtain additional expressions
    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    #Update inner_vars
    inner_vars[PSID.P_ES_var] = p_elec_out

    #Update states
    device_states[gfm_indices[:states][:θ_oc]] = θ0_oc #θ_oc 
    device_states[gfm_indices[:states][:p_oc]] = p_elec_out #pm
    device_states[gfm_indices[:states][:q_oc]] = q_elec_out #qm

    #Update inner vars
    inner_vars[PSID.θ_oc_var] = θ0_oc
    inner_vars[PSID.ω_oc_var] = 1.0 #get_ω_ref(dynamic_device)
    #Update Q_ref. Initialization assumes q_ref = q_elec_out of PF solution
    references[gfm_indices[:references][:P_ref]] = p_elec_out
    references[gfm_indices[:references][:Q_ref]] = q_elec_out
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
    s::Union{GFM, PSIDS.GFMParams},
)
    inner_vars[PSID.Vdc_var] = params[gfm_indices[:params][:voltage_gfm]]
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
    s::Union{GFM, PSIDS.GFMParams},
)

    #Obtain external states inputs for component
    Ir_filter = device_states[gfm_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfm_indices[:states][:ii_filter]]
    Ir_cnv = device_states[gfm_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfm_indices[:states][:ii_cnv]]
    Vr_filter = device_states[gfm_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfm_indices[:states][:vi_filter]]

    #Obtain inner variables for component
    ω_oc = 1.0 #get_ω_ref(dynamic_device)
    θ0_oc = inner_vars[PSID.θ_oc_var]
    Vdc = inner_vars[PSID.Vdc_var]

    #Obtain output of converter
    Vr_cnv0 = inner_vars[PSID.Vr_cnv_var]
    Vi_cnv0 = inner_vars[PSID.Vi_cnv_var]

    #Get Voltage Controller parameters
    kpv = params[gfm_indices[:params][:kpv]]
    kiv = params[gfm_indices[:params][:kiv]]
    kffi = params[gfm_indices[:params][:kffi]]
    cf = params[gfm_indices[:params][:cf_gfm]]
    rv = params[gfm_indices[:params][:rv]]
    lv = params[gfm_indices[:params][:lv]]
    kpc = params[gfm_indices[:params][:kpc_gfm]]
    kic = params[gfm_indices[:params][:kic_gfm]]
    kffv = params[gfm_indices[:params][:kffv_gfm]]
    lf = params[gfm_indices[:params][:lf_gfm]]
    ωad = params[gfm_indices[:params][:ωad]]
    kad = params[gfm_indices[:params][:kad]]

    function f!(out, x, params, t)
        θ_oc = x[1]
        v_refr = x[2]
        ξ_d = x[3]
        ξ_q = x[4]
        γ_d = x[5]
        γ_q = x[6]
        ϕ_d = x[7]
        ϕ_q = x[8]

        #Reference Frame Transformations
        I_dq_filter = PSID.ri_dq(θ_oc + pi / 2) * [Ir_filter; Ii_filter]
        I_dq_cnv = PSID.ri_dq(θ_oc + pi / 2) * [Ir_cnv; Ii_cnv]
        V_dq_filter = PSID.ri_dq(θ_oc + pi / 2) * [Vr_filter; Vi_filter]
        V_dq_cnv0 = PSID.ri_dq(θ_oc + pi / 2) * [Vr_cnv0; Vi_cnv0]

        #Voltage controller references
        Vd_filter_ref =
            (v_refr - rv * I_dq_filter[PSID.d] + ω_oc * lv * I_dq_filter[PSID.q])
        Vq_filter_ref = (-rv * I_dq_filter[PSID.q] - ω_oc * lv * I_dq_filter[PSID.d])

        #Current controller references
        Id_cnv_ref = (
            kpv * (Vd_filter_ref - V_dq_filter[PSID.d]) + kiv * ξ_d -
            cf * ω_oc * V_dq_filter[PSID.q] + kffi * I_dq_filter[PSID.d]
        )
        Iq_cnv_ref = (
            kpv * (Vq_filter_ref - V_dq_filter[PSID.q]) +
            kiv * ξ_q +
            cf * ω_oc * V_dq_filter[PSID.d] +
            kffi * I_dq_filter[PSID.q]
        )

        #References for Converter Output Voltage
        Vd_cnv_ref = (
            kpc * (Id_cnv_ref - I_dq_cnv[PSID.d]) + kic * γ_d -
            ω_oc * lf * I_dq_cnv[PSID.q] + kffv * V_dq_filter[PSID.d] -
            kad * (V_dq_filter[PSID.d] - ϕ_d)
        )
        Vq_cnv_ref = (
            kpc * (Iq_cnv_ref - I_dq_cnv[PSID.q]) +
            kic * γ_q +
            ω_oc * lf * I_dq_cnv[PSID.d] +
            kffv * V_dq_filter[PSID.q] - kad * (V_dq_filter[PSID.q] - ϕ_q)
        )

        out[1] = Vd_filter_ref - V_dq_filter[PSID.d]
        out[2] = Vq_filter_ref - V_dq_filter[PSID.q]
        out[3] = Id_cnv_ref - I_dq_cnv[PSID.d]
        out[4] = Iq_cnv_ref - I_dq_cnv[PSID.q]
        out[5] = V_dq_filter[PSID.d] - ϕ_d
        out[6] = V_dq_filter[PSID.q] - ϕ_q
        out[7] = Vd_cnv_ref - V_dq_cnv0[PSID.d]
        out[8] = Vq_cnv_ref - V_dq_cnv0[PSID.q]
    end
    x0 = [θ0_oc, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #sol = NLsolve.nlsolve(f!, x0, ftol = STRICT_NLSOLVE_F_TOLERANCE)
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
        #Update angle:
        inner_vars[PSID.θ_oc_var] = sol_x0[1]
        #Assumes that angle is in second position
        device_states[gfm_indices[:states][:θ_oc]] = sol_x0[1]
        inner_vars[PSID.θ_oc_var] = sol_x0[1]
        references[gfm_indices[:references][:V_ref]] = sol_x0[2]
        inner_vars[PSID.V_oc_var] = sol_x0[2]

        #Update Converter modulation
        m0_dq = (PSID.ri_dq(sol_x0[1] + pi / 2) * [Vr_cnv0; Vi_cnv0]) ./ Vdc
        inner_vars[PSID.md_var] = m0_dq[PSID.d]
        inner_vars[PSID.mq_var] = m0_dq[PSID.q]
        #Update states
        device_states[gfm_indices[:states][:ξd_ic]] = sol_x0[3] #ξ_d
        device_states[gfm_indices[:states][:ξq_ic]] = sol_x0[4] #ξ_q
        device_states[gfm_indices[:states][:γd_ic]] = sol_x0[5] #γ_d
        device_states[gfm_indices[:states][:γq_ic]] = sol_x0[6] #γ_q
        device_states[gfm_indices[:states][:ϕd_ic]] = sol_x0[7] #ϕ_d
        device_states[gfm_indices[:states][:ϕq_ic]] = sol_x0[8] #ϕ_q
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
    s::Union{GFM, PSIDS.GFMParams},
)
    #Obtain global vars
    sys_ω = 1.0 #global_vars[GLOBAL_VAR_SYS_FREQ_INDEX]

    #Update Voltage data
    inner_vars[PSID.Vr_inv_var] = Vr#V(t)[1]
    inner_vars[PSID.Vi_inv_var] = Vi#V(t)[2]

    #Is Vref used in GFM equations?
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
        s::Union{GFM, PSIDS.GFMParams},
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
        s::Union{GFM, PSIDS.GFMParams},
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
        s::Union{GFM, PSIDS.GFMParams},
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
        s::Union{GFM, PSIDS.GFMParams},
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
        s::Union{GFM, PSIDS.GFMParams},
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
        s::Union{GFM, PSIDS.GFMParams},
    )

    mdl_filter_ode!(
        output_ode,
        device_states,
        p,
        references,
        inner_vars,
        Vr,
        Vi,
        s::Union{GFM, PSIDS.GFMParams},
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
    s::Union{GFM, PSIDS.GFMParams},
) end

function mdl_DCside_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFM, PSIDS.GFMParams},
)
    #Update inner_vars
    inner_vars[PSID.Vdc_var] = p[gfm_indices[:params][:voltage_gfm]]
end

function mdl_freq_estimator_ode!(
    output_ode,
    device_states,
    p,
    references,
    inner_vars,
    Vr,
    Vi,
    s::Union{GFM, PSIDS.GFMParams},
)
    #Update inner_vars
    #PLL frequency
    inner_vars[PSID.ω_freq_estimator_var] = 1.0 #frequency
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
    s::Union{GFM, PSIDS.GFMParams},
)
    Vr_filter = device_states[gfm_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfm_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfm_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfm_indices[:states][:ii_filter]]

    #Get Active Power Controller parameters
    Rp = p[gfm_indices[:params][:Rp]]
    ωz = p[gfm_indices[:params][:ωz_gfm]]
    ωb = 2 * pi * 60.0 #Rated angular frequency

    #Get Reactive Power Controller parameters
    kq = p[gfm_indices[:params][:kq]]
    ωf = p[gfm_indices[:params][:ωf_gfm]]

    #Obtain external parameters
    p_ref = references[gfm_indices[:references][:P_ref]]
    ω_ref = 1.0
    V_ref = references[gfm_indices[:references][:V_ref]]
    q_ref = references[gfm_indices[:references][:Q_ref]]

    #Define internal states for outer control
    θ_oc = device_states[gfm_indices[:states][:θ_oc]]
    pm = device_states[gfm_indices[:states][:p_oc]]
    qm = device_states[gfm_indices[:states][:q_oc]]

    #Obtain additional expressions
    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    #Compute Frequency from Droop
    ω_oc = ω_ref + Rp * (p_ref - pm)

    #Compute block derivatives
    _, dpm_dt = PSID.low_pass(p_elec_out, pm, 1.0, 1.0 / ωz)
    _, dqm_dt = PSID.low_pass(q_elec_out, qm, 1.0, 1.0 / ωf)

    #ext = PSY.get_ext(outer_control)
    #bool_val = get(ext, "is_not_reference", 1.0)
    bool_val = 1.0 #assume not reference
    ω_sys = 1.0 #assume fixed frequency

    #Compute 3 states ODEs
    output_ode[gfm_indices[:states][:θ_oc]] = bool_val * ωb * (ω_oc - ω_sys)
    output_ode[gfm_indices[:states][:p_oc]] = dpm_dt
    output_ode[gfm_indices[:states][:q_oc]] = dqm_dt

    #Update inner vars
    inner_vars[PSID.θ_oc_var] = θ_oc
    inner_vars[PSID.ω_oc_var] = ω_oc
    inner_vars[PSID.V_oc_var] = V_ref + kq * (q_ref - qm)
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
    s::Union{GFM, PSIDS.GFMParams},
)
    Ir_filter = device_states[gfm_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfm_indices[:states][:ii_filter]]
    Ir_cnv = device_states[gfm_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfm_indices[:states][:ii_cnv]]
    Vr_filter = device_states[gfm_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfm_indices[:states][:vi_filter]]

    #Obtain inner variables for component
    ω_oc = inner_vars[PSID.ω_oc_var]
    θ_oc = inner_vars[PSID.θ_oc_var]
    v_refr = inner_vars[PSID.V_oc_var]
    Vdc = inner_vars[PSID.Vdc_var]

    #Get Voltage Controller parameters
    kpv = p[gfm_indices[:params][:kpv]]
    kiv = p[gfm_indices[:params][:kiv]]
    kffi = p[gfm_indices[:params][:kffi]]
    cf = p[gfm_indices[:params][:cf_gfm]]
    rv = p[gfm_indices[:params][:rv]]
    lv = p[gfm_indices[:params][:lv]]
    kpc = p[gfm_indices[:params][:kpc_gfm]]
    kic = p[gfm_indices[:params][:kic_gfm]]
    kffv = p[gfm_indices[:params][:kffv_gfm]]
    lf = p[gfm_indices[:params][:lf_gfm]]
    ωad = p[gfm_indices[:params][:ωad]]
    kad = p[gfm_indices[:params][:kad]]

    #Define internal states for frequency estimator
    ξ_d = device_states[gfm_indices[:states][:ξd_ic]]
    ξ_q = device_states[gfm_indices[:states][:ξq_ic]]
    γ_d = device_states[gfm_indices[:states][:γd_ic]]
    γ_q = device_states[gfm_indices[:states][:γq_ic]]
    ϕ_d = device_states[gfm_indices[:states][:ϕd_ic]]
    ϕ_q = device_states[gfm_indices[:states][:ϕq_ic]]

    #Transformations to dq frame
    I_dq_filter = PSID.ri_dq(θ_oc + pi / 2) * [Ir_filter; Ii_filter]
    I_dq_cnv = PSID.ri_dq(θ_oc + pi / 2) * [Ir_cnv; Ii_cnv]
    V_dq_filter = PSID.ri_dq(θ_oc + pi / 2) * [Vr_filter; Vi_filter]

    ### Compute 6 states ODEs (D'Arco EPSR122 Model) ###
    ## SRF Voltage Control w/ Virtual Impedance ##
    #Virtual Impedance
    Vd_filter_ref = (v_refr - rv * I_dq_filter[PSID.d] + ω_oc * lv * I_dq_filter[PSID.q])
    Vq_filter_ref = (-rv * I_dq_filter[PSID.q] - ω_oc * lv * I_dq_filter[PSID.d])

    #Voltage Control PI Blocks
    Id_pi, dξd_dt = PSID.pi_block(Vd_filter_ref - V_dq_filter[PSID.d], ξ_d, kpv, kiv)
    Iq_pi, dξq_dt = PSID.pi_block(Vq_filter_ref - V_dq_filter[PSID.q], ξ_q, kpv, kiv)
    #PI Integrator (internal state)
    output_ode[gfm_indices[:states][:ξd_ic]] = dξd_dt
    output_ode[gfm_indices[:states][:ξq_ic]] = dξq_dt

    #Compensate output Control Signal - Links to SRF Current Controller
    Id_cnv_ref = Id_pi - cf * ω_oc * V_dq_filter[PSID.q] + kffi * I_dq_filter[PSID.d]
    Iq_cnv_ref = Iq_pi + cf * ω_oc * V_dq_filter[PSID.d] + kffi * I_dq_filter[PSID.q]

    ## SRF Current Control ##
    #Current Control PI Blocks
    Vd_pi, dγd_dt = PSID.pi_block(Id_cnv_ref - I_dq_cnv[PSID.d], γ_d, kpc, kic)
    Vq_pi, dγq_dt = PSID.pi_block(Iq_cnv_ref - I_dq_cnv[PSID.q], γ_q, kpc, kic)
    #PI Integrator (internal state)
    output_ode[gfm_indices[:states][:γd_ic]] = dγd_dt
    output_ode[gfm_indices[:states][:γq_ic]] = dγq_dt

    #Compensate References for Converter Output Voltage
    Vd_cnv_ref =
        Vd_pi - ω_oc * lf * I_dq_cnv[PSID.q] + kffv * V_dq_filter[PSID.d] -
        kad * (V_dq_filter[PSID.d] - ϕ_d)
    Vq_cnv_ref =
        Vq_pi + ω_oc * lf * I_dq_cnv[PSID.d] + kffv * V_dq_filter[PSID.q] -
        kad * (V_dq_filter[PSID.q] - ϕ_q)

    #Active Damping LPF (internal state)
    output_ode[gfm_indices[:states][:ϕd_ic]] =
        PSID.low_pass(V_dq_filter[PSID.d], ϕ_d, 1.0, 1.0 / ωad)[2]
    output_ode[gfm_indices[:states][:ϕq_ic]] =
        PSID.low_pass(V_dq_filter[PSID.q], ϕ_q, 1.0, 1.0 / ωad)[2]

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
    s::Union{GFM, PSIDS.GFMParams},
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
    s::Union{GFM, PSIDS.GFMParams},
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
    lf = p[gfm_indices[:params][:lf_gfm]]
    rf = p[gfm_indices[:params][:rf_gfm]]
    cf = p[gfm_indices[:params][:cf_gfm]]
    lg = p[gfm_indices[:params][:lg_gfm]]
    rg = p[gfm_indices[:params][:rg_gfm]]

    #Define internal states for filter
    Ir_cnv = device_states[gfm_indices[:states][:ir_cnv]]
    Ii_cnv = device_states[gfm_indices[:states][:ii_cnv]]
    Vr_filter = device_states[gfm_indices[:states][:vr_filter]]
    Vi_filter = device_states[gfm_indices[:states][:vi_filter]]
    Ir_filter = device_states[gfm_indices[:states][:ir_filter]]
    Ii_filter = device_states[gfm_indices[:states][:ii_filter]]

    #Inputs (control signals) - N/A

    #Compute 6 states ODEs (D'Arco EPSR122 Model)
    #Inverter Output Inductor (internal state)
    #𝜕id_c/𝜕t
    output_ode[gfm_indices[:states][:ir_cnv]] =
        (ωb / lf) * (Vr_cnv - Vr_filter - rf * Ir_cnv + lf * ω_sys * Ii_cnv)
    #𝜕iq_c/𝜕t
    output_ode[gfm_indices[:states][:ii_cnv]] =
        (ωb / lf) * (Vi_cnv - Vi_filter - rf * Ii_cnv - lf * ω_sys * Ir_cnv)
    #LCL Capacitor (internal state)
    #𝜕vd_o/𝜕t
    output_ode[gfm_indices[:states][:vr_filter]] =
        (ωb / cf) * (Ir_cnv - Ir_filter + cf * ω_sys * Vi_filter)
    #𝜕vq_o/𝜕t
    output_ode[gfm_indices[:states][:vi_filter]] =
        (ωb / cf) * (Ii_cnv - Ii_filter - cf * ω_sys * Vr_filter)
    #Grid Inductance (internal state)
    #𝜕id_o/𝜕t
    output_ode[gfm_indices[:states][:ir_filter]] =
        (ωb / lg) * (Vr_filter - V_tR - rg * Ir_filter + lg * ω_sys * Ii_filter)
    #𝜕iq_o/𝜕t
    output_ode[gfm_indices[:states][:ii_filter]] =
        (ωb / lg) * (Vi_filter - V_tI - rg * Ii_filter - lg * ω_sys * Ir_filter)

    #Update inner_vars
    inner_vars[PSID.Vr_filter_var] = Vr_filter
    inner_vars[PSID.Vi_filter_var] = Vi_filter
    return
end
