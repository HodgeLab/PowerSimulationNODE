mutable struct partitioned_θ
    θ_node::Vector{}
    θ_u0::Vector{}
    θ_observation::Vector{}
end

function partitioned_θ()
    return partitioned_θ(Float64[], Float64[], Float64[])
end

mutable struct θ_lengths
    node::Int64
    u0::Int64
    observation::Int64
end

function split_θ(θ, lengths::θ_lengths)
    @assert length(θ) == (lengths.node + lengths.u0 + lengths.observation)
    range_node = 1:(lengths.node)
    range_u0 = (lengths.node + 1):(lengths.node + lengths.u0)
    range_observation =
        ((lengths.node + lengths.u0) + 1):(lengths.node + lengths.u0 + lengths.observation)
    return partitioned_θ(θ[range_node], θ[range_u0], θ[range_observation])
end

function combine_θ(θ::partitioned_θ)
    return vcat(θ.θ_node, θ.θ_u0, θ.θ_observation),
    θ_lengths(length(θ.θ_node), length(θ.θ_u0), length(θ.θ_observation))
end

function split_θ_u0(θ_u0::Vector{}, pvs_index::Int64, n_pvs::Int64)
    length_subset = Int(length(θ_u0) / n_pvs)
    return θ_u0[((pvs_index - 1) * length_subset + 1):((pvs_index) * length_subset)]
end
