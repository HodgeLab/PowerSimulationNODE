"""
    mutable struct NeuralODE <: DynamicInjection
        name::String
        n_states::Int
        n_hidden_layers::Int
        n_neurons::Int
        activation_function::String
        nn_parameters::Vector{Float64}
        observation_function::String
        observation_parameters::Vector{Float64}
        base_power::Float64
        states::Vector{Symbol}
        ext::Dict{String, Any}
        internal::InfrastructureSystemsInternal
    end

Densely connected NODE with variable number of feedback states.

# Arguments
- `name::String`
- `n_states::Int`: Total number of states in NeuralODE
- `n_hidden_layers::Int`: Number of hidden layers
- `n_neurons::Int`: Neurons per hidden layer
- `activation_function::String`: Activation function for input and hidden layers. Output layer has identity activation.
- `nn_parameters::Vector{Float64}`: Weights and biases of the nn which defines the dynamics of the component
- `observation_function::String`: Observation function from latent space of NODE to observable space
- `observation_parameters::Vector{Float64}`: Parameters of the observation function
- `base_power::Float64`: Base power
- `states::Vector{Symbol}`: Real output current, imaginary output current
- `ext::Dict{String, Any}`
- `internal::InfrastructureSystemsInternal`: power system internal reference, do not modify
"""
mutable struct NeuralODE <: DynamicInjection
    name::String
    "Total number of states in NeuralODE"
    n_states::Int
    "Number of hidden layers"
    n_hidden_layers::Int
    "Neurons per hidden layer"
    n_neurons::Int
    "Activation function for input and hidden layers. Output layer has identity activation."
    activation_function::String
    "Weights and biases of the nn which defines the dynamics of the component"
    nn_parameters::Vector{Float64}
    "Observation function from latent space of NODE to observable space"
    observation_function::String
    "Parameters of the observation function"
    observation_parameters::Vector{Float64}
    "Base power"
    base_power::Float64
    "Real output current, imaginary output current"
    states::Vector{Symbol}
    ext::Dict{String, Any}
    "power system internal reference, do not modify"
    internal::InfrastructureSystemsInternal
end

function NeuralODE(name, n_hidden_layers, n_neurons, activation_function, nn_parameters, observation_function, observation_parameters, base_power=100.0, ext=Dict{String, Any}(), )
    NeuralODE(name, n_hidden_layers, n_neurons, activation_function, nn_parameters, observation_function, observation_parameters, base_power, ext, 2, [:Ir, :Ii], InfrastructureSystemsInternal(), )
end

function NeuralODE(; name, n_states=2, n_hidden_layers, n_neurons, activation_function, nn_parameters, observation_function, observation_parameters, base_power=100.0, states=[:Ir, :Ii], ext=Dict{String, Any}(), internal=InfrastructureSystemsInternal(), )
    NeuralODE(name, n_states, n_hidden_layers, n_neurons, activation_function, nn_parameters, observation_function, observation_parameters, base_power, states, ext, internal, )
end

# Constructor for demo purposes; non-functional.
function NeuralODE(::Nothing)
    NeuralODE(;
        name="init",
        n_hidden_layers=0,
        n_neurons=0,
        activation_function="0",
        nn_parameters=Any[0],
        observation_function="0",
        observation_parameters=Any[0],
        base_power=0,
        ext=Dict{String, Any}(),
    )
end

"""Get [`NeuralODE`](@ref) `name`."""
get_name(value::NeuralODE) = value.name
"""Get [`NeuralODE`](@ref) `n_states`."""
get_n_states(value::NeuralODE) = value.n_states
"""Get [`NeuralODE`](@ref) `n_hidden_layers`."""
get_n_hidden_layers(value::NeuralODE) = value.n_hidden_layers
"""Get [`NeuralODE`](@ref) `n_neurons`."""
get_n_neurons(value::NeuralODE) = value.n_neurons
"""Get [`NeuralODE`](@ref) `activation_function`."""
get_activation_function(value::NeuralODE) = value.activation_function
"""Get [`NeuralODE`](@ref) `nn_parameters`."""
get_nn_parameters(value::NeuralODE) = value.nn_parameters
"""Get [`NeuralODE`](@ref) `observation_function`."""
get_observation_function(value::NeuralODE) = value.observation_function
"""Get [`NeuralODE`](@ref) `observation_parameters`."""
get_observation_parameters(value::NeuralODE) = value.observation_parameters
"""Get [`NeuralODE`](@ref) `base_power`."""
get_base_power(value::NeuralODE) = value.base_power
"""Get [`NeuralODE`](@ref) `states`."""
get_states(value::NeuralODE) = value.states
"""Get [`NeuralODE`](@ref) `ext`."""
get_ext(value::NeuralODE) = value.ext
"""Get [`NeuralODE`](@ref) `internal`."""
get_internal(value::NeuralODE) = value.internal

"""Set [`NeuralODE`](@ref) `n_states`."""
set_n_states!(value::NeuralODE, val) = value.n_states = val
"""Set [`NeuralODE`](@ref) `n_hidden_layers`."""
set_n_hidden_layers!(value::NeuralODE, val) = value.n_hidden_layers = val
"""Set [`NeuralODE`](@ref) `n_neurons`."""
set_n_neurons!(value::NeuralODE, val) = value.n_neurons = val
"""Set [`NeuralODE`](@ref) `activation_function`."""
set_activation_function!(value::NeuralODE, val) = value.activation_function = val
"""Set [`NeuralODE`](@ref) `nn_parameters`."""
set_nn_parameters!(value::NeuralODE, val) = value.nn_parameters = val
"""Set [`NeuralODE`](@ref) `observation_function`."""
set_observation_function!(value::NeuralODE, val) = value.observation_function = val
"""Set [`NeuralODE`](@ref) `observation_parameters`."""
set_observation_parameters!(value::NeuralODE, val) = value.observation_parameters = val
"""Set [`NeuralODE`](@ref) `base_power`."""
set_base_power!(value::NeuralODE, val) = value.base_power = val
"""Set [`NeuralODE`](@ref) `ext`."""
set_ext!(value::NeuralODE, val) = value.ext = val

