#= zip
Recipe for adding new surrogate model from PSID model.
    * Copy individual initialization and device functions.
    * Modify any parameters or values that come from PSID devices.
    * Add non-zero mass matrix time constants to the appropriate RHS equations. 
    * Change powerflow devices to be derived from v0, i0, not quantities from the device. 
    * Make a constant dictionary (e.g. zip_indices) with all of the location indices within the vectors (constant per device).
 =#

using Flux
abstract type ZIPLayer <: Function end
basic_tgrad(du, u, p, t) = zero(u)
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
            p = Float64[
                1 / 3,  # P_Z weight 
                1 / 3,  # P_I weight 
                1 / 3,  # P_P weight 
                1 / 3,  # Q_Z weight 
                1 / 3,  # Q_I weight 
                1 / 3,  # Q_P weight 
            ]
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

    i_series = Array{Float64}(undef, 2, length(tsteps))
    for i in eachindex(tsteps)
        i_series[1, i], i_series[2, i] = device(v0, i0, p_ordered, V, tsteps[i], s::ZIP)
    end
    return PhysicalModel_solution(tsteps, i_series, [], true)
end

#Note: got rid of types to work with ForwardDiff
struct PhysicalModel_solution
    t_series::Any
    i_series::Any
    res::Any
    converged::Bool
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

function device(v0, i0, p, V, t, s::ZIP)
    Ir, Ii = mdl_zip_load(v0, i0, p, V, t, s::ZIP)  #takes
    return Ir, Ii
end

function mdl_zip_load(v0, i0, p, V, t, s::ZIP)
    # Read power flow voltages
    #V0_mag_inv = 1.0 / get_V_ref(wrapper)
    V0_mag_inv = 1.0 / sqrt(v0[1]^2 + v0[2]^2) # PSY.get_magnitude(PSY.get_bus(wrapper))
    V0_mag_sq_inv = V0_mag_inv^2
    S0_total = (v0[1] + im * v0[2]) * conj(i0[1] + im * i0[2])
    P0_total = real(S0_total)
    Q0_total = imag(S0_total)

    voltage_r = V(t)[1]
    voltage_i = V(t)[2]
    V_mag = sqrt(voltage_r^2 + voltage_i^2)
    V_mag_inv = 1.0 / V_mag
    V_mag_sq_inv = V_mag_inv^2

    # Load device parameters
    max_active_power_Z = p[zip_indices[:params][:max_active_power_Z]]
    max_active_power_I = p[zip_indices[:params][:max_active_power_I]]
    max_active_power_P = p[zip_indices[:params][:max_active_power_P]]
    P_base_total = max_active_power_Z + max_active_power_I + max_active_power_P
    max_reactive_power_Z = p[zip_indices[:params][:max_reactive_power_Z]]
    max_reactive_power_I = p[zip_indices[:params][:max_reactive_power_I]]
    max_reactive_power_P = p[zip_indices[:params][:max_reactive_power_P]]
    Q_base_total = max_reactive_power_Z + max_reactive_power_I + max_reactive_power_P

    P_power = (P0_total / P_base_total) * max_active_power_P
    P_current = (P0_total / P_base_total) * max_active_power_I
    P_impedance = (P0_total / P_base_total) * max_active_power_Z
    Q_power = (Q0_total / Q_base_total) * max_reactive_power_P
    Q_current = (Q0_total / Q_base_total) * max_reactive_power_I
    Q_impedance = (Q0_total / Q_base_total) * max_reactive_power_Z

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
