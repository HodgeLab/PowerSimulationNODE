#=
Recipe for adding new surrogate model from PSID model.
    * Copy individual initialization and device functions.
    * Modify any parameters or values that come from PSID devices.
    * Add non-zero mass matrix time constants to the appropriate RHS equations. 
    * Change powerflow devices to be derived from v0, i0, not quantities from the device. 
    * Make a constant dictionary (e.g. zip_indices) with all of the location indices within the vectors (constant per device).
 =#
#TODO - what does passing 0 as time gradient do - from Avik's code
basic_tgrad(u, p, t) = zero(u)
basic_tgrad_inplace(du, u, p, t) = zero(u)

#Note: got rid of types to work with ForwardDiff
struct PhysicalModel_solution
    t_series::Any
    i_series::Any
    res::Any
    converged::Bool
end

struct SteadyStateNeuralODE_solution{T}
    r0_pred::AbstractArray{T}
    r0::AbstractArray{T}
    deq_iterations::Int64
    ssstats::Union{SciMLBase.NLStats, Nothing}
    t_series::AbstractArray{T}
    r_series::AbstractArray{T}
    i_series::AbstractArray{T}
    res::AbstractArray{T}
    converged::Bool
    destats::Union{DiffEqBase.DEStats, Nothing}
end
