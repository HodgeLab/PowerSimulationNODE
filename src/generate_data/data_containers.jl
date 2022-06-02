struct TrainData{T}
    tsteps::AbstractArray{T}
    groundtruth_current::AbstractArray{T}
    connecting_impedance::AbstractArray{T}
    powerflow::AbstractArray{T}
end

mutable struct PVSData
    internal_voltage_bias::Float64
    internal_voltage_frequencies::Vector{Float64}
    internal_voltage_coefficients::Vector{Tuple{Float64, Float64}}
    internal_angle_bias::Float64
    internal_angle_frequencies::Vector{Float64}
    internal_angle_coefficients::Vector{Tuple{Float64, Float64}}
    V0::Float64
    θ0::Float64
    P0::Float64
    Q0::Float64
    branch_name::String
    from_or_to::String
end

struct SurrogateTrainInputs
    branch_order::Vector{String}
    train_data::Vector{TrainData}
end
