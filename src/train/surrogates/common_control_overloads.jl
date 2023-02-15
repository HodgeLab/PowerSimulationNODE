#Overload PSID common control methods to allow for parameters of dual types during training. 
"""
Low Pass Filter Modified
     ┌─────────────┐
     │      K      │
u -> │ ────────────│ -> y
     │ K_den + sT  │
     └─────────────┘
"""
function PSID.low_pass_modified_mass_matrix(
    u::Z,
    y::V,
    K::Float64,
    K_den::W,
    ::X,
) where {
    V <: PSID.ACCEPTED_REAL_TYPES,
    W <: PSID.ACCEPTED_REAL_TYPES,
    Z <: PSID.ACCEPTED_REAL_TYPES,
    X <: PSID.ACCEPTED_REAL_TYPES,
}
    return y, K * u - K_den * y
end

function PSID.low_pass_modified(
    u::Z,
    y::V,
    K::Float64,
    K_den::W,
    T::X,
) where {
    V <: PSID.ACCEPTED_REAL_TYPES,
    W <: PSID.ACCEPTED_REAL_TYPES,
    Z <: PSID.ACCEPTED_REAL_TYPES,
    X <: PSID.ACCEPTED_REAL_TYPES,
}
    return y, (1.0 / T) * PSID.low_pass_modified_mass_matrix(u, y, K, K_den, T)[2]
end

"""
Low Pass Filter
     ┌────────┐
     │    K   │
u -> │ ────── │ -> y
     │ 1 + sT │
     └────────┘
"""

# Use this one if T = 0 is allowed, and let the mass matrix take care of it.
function PSID.low_pass_mass_matrix(
    u::Z,
    y::V,
    K::Float64,
    T::W,
) where {
    V <: PSID.ACCEPTED_REAL_TYPES,
    Z <: PSID.ACCEPTED_REAL_TYPES,
    W <: PSID.ACCEPTED_REAL_TYPES,
}
    return PSID.low_pass_modified_mass_matrix(u, y, K, 1.0, T)
end

function PSID.low_pass(
    u::Z,
    y::V,
    K::Float64,
    T::W,
) where {
    V <: PSID.ACCEPTED_REAL_TYPES,
    Z <: PSID.ACCEPTED_REAL_TYPES,
    W <: PSID.ACCEPTED_REAL_TYPES,
}
    return PSID.low_pass_modified(u, y, K, 1.0, T)
end

"""
Proportional-Integral Block
             y_max
            /¯¯¯¯¯¯
     ┌──────────┐
     │      ki  │
u -> │kp + ───  │ -> y
     │      s   │
     └──────────┘
    ______/
     y_min

Internal State: x
"""

function PSID.pi_block(
    u::Z,
    x::V,
    kp::W,
    ki::X,
) where {
    Z <: PSID.ACCEPTED_REAL_TYPES,
    V <: PSID.ACCEPTED_REAL_TYPES,
    W <: PSID.ACCEPTED_REAL_TYPES,
    X <: PSID.ACCEPTED_REAL_TYPES,
}
    return kp * u + ki * x, u
end
