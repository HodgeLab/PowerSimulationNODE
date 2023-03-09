
using Flux
abstract type ZIPLayer <: Function end
Flux.trainable(m::ZIPLayer) = (p = m.p,)

struct ZIP{PT, PF, PM, SS, DS, A, K} <: ZIPLayer
    p_train::PT
    p_fixed::PF
    p_map::PM
    ss_solver::SS
    dyn_solver::DS
    args::A
    kwargs::K

    function ZIP(  #This is an inner constructor 
        ss_solver,
        dyn_solver,
        args...;
        p = nothing,
        kwargs...,
    )
        if p === nothing
            p = default_params(PSIDS.ZIPParams())
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

Flux.@functor ZIP
Flux.trainable(m::ZIP) = (p = m.p_train,)

function (s::ZIP)(
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
    refs = zeros(typeof(p_ordered[1]), n_refs(s))
    initilize_static_device!(refs, p, v0, i0)
    i_series = Array{Float64}(undef, 2, length(tsteps))
    for i in eachindex(tsteps)
        Vr, Vi = V(tsteps[i])
        i_series[1, i], i_series[2, i] = device(p_ordered, refs, Vr, Vi, s)
    end
    return PhysicalModel_solution(tsteps, i_series, [1.0], true)
end

function default_params(::PSIDS.ZIPParams)
    return Float64[1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]
end

function ordered_param_symbols(::Union{ZIP, PSIDS.ZIPParams})
    return [
        :max_active_power_Z,
        :max_active_power_I,
        :max_active_power_P,
        :max_reactive_power_Z,
        :max_reactive_power_I,
        :max_reactive_power_P,
    ]
end

function n_refs(::Union{ZIP, PSIDS.ZIPParams})
    return 8
end

function n_params(::Union{ZIP, PSIDS.ZIPParams})
    return 6
end

const zip_indices = Dict{Symbol, Dict{Symbol, Int64}}(
    :params => Dict{Symbol, Int64}(
        :max_active_power_Z => 1,
        :max_active_power_I => 2,
        :max_active_power_P => 3,
        :max_reactive_power_Z => 4,
        :max_reactive_power_I => 5,
        :max_reactive_power_P => 6,
    ),
    :states => Dict{Symbol, Int64}(),
    :references => Dict{Symbol, Int64}(),
)

function initilize_static_device!(references, p, v0, i0)
    S0_total = (v0[1] + im * v0[2]) * conj(i0[1] + im * i0[2])
    P0_total = real(S0_total)
    Q0_total = imag(S0_total)

    # Load device parameters
    max_active_power_Z = p[zip_indices[:params][:max_active_power_Z]]
    max_active_power_I = p[zip_indices[:params][:max_active_power_I]]
    max_active_power_P = p[zip_indices[:params][:max_active_power_P]]
    P_base_total = max_active_power_Z + max_active_power_I + max_active_power_P
    max_reactive_power_Z = p[zip_indices[:params][:max_reactive_power_Z]]
    max_reactive_power_I = p[zip_indices[:params][:max_reactive_power_I]]
    max_reactive_power_P = p[zip_indices[:params][:max_reactive_power_P]]
    Q_base_total = max_reactive_power_Z + max_reactive_power_I + max_reactive_power_P

    references[1] = v0[1]
    references[2] = v0[2]
    references[3] = (P0_total / P_base_total) * max_active_power_P
    references[4] = (P0_total / P_base_total) * max_active_power_I
    references[5] = (P0_total / P_base_total) * max_active_power_Z
    references[6] = (Q0_total / Q_base_total) * max_reactive_power_P
    references[7] = (Q0_total / Q_base_total) * max_reactive_power_I
    references[8] = (Q0_total / Q_base_total) * max_reactive_power_Z
end
function device(p_ordered, references, Vr, Vi, s::Union{ZIP, PSIDS.ZIPParams})
    Ir, Ii = mdl_zip_load(p_ordered, references, Vr, Vi, s)
    return Ir, Ii
end

function mdl_zip_load(
    p_ordered,
    references,
    voltage_r,
    voltage_i,
    s::Union{ZIP, PSIDS.ZIPParams},
)
    Vr0 = references[1]
    Vi0 = references[2]
    V0_mag_inv = 1.0 / sqrt(Vr0^2 + Vi0^2)
    V0_mag_sq_inv = V0_mag_inv^2

    V_mag = sqrt(voltage_r^2 + voltage_i^2)
    V_mag_inv = 1.0 / V_mag
    V_mag_sq_inv = V_mag_inv^2

    P_power = references[3]
    P_current = references[4]
    P_impedance = references[5]
    Q_power = references[6]
    Q_current = references[7]
    Q_impedance = references[8]
    # Compute ZIP currents
    Iz_re = V0_mag_sq_inv * (voltage_r * P_impedance + voltage_i * Q_impedance)
    Iz_im = V0_mag_sq_inv * (voltage_i * P_impedance - voltage_r * Q_impedance)

    Ii_re = V0_mag_inv * V_mag_inv * (voltage_r * P_current + voltage_i * Q_current)
    Ii_im = V0_mag_inv * V_mag_inv * (voltage_i * P_current - voltage_r * Q_current)

    Ip_re = V_mag_sq_inv * (voltage_r * P_power + voltage_i * Q_power)
    Ip_im = V_mag_sq_inv * (voltage_i * P_power - voltage_r * Q_power)

    current_r = -(Iz_re + Ii_re + Ip_re) #in system pu flowing out
    current_i = -(Iz_im + Ii_im + Ip_im) #in system pu flowing out 

    return current_r, current_i
end
