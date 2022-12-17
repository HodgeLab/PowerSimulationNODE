
abstract type SurrogateModelParams end

struct SteadyStateNODEParams <: SurrogateModelParams
    type::String
    initializer_layer_type::String
    initializer_n_layer::Int64
    initializer_width_layers::Int64
    initializer_activation::String
    dynamic_layer_type::String
    dynamic_hidden_states::Int64
    dynamic_n_layer::Int64
    dynamic_width_layers::Int64
    dynamic_activation::String
    dynamic_σ2_initialization::Float64
end

function SteadyStateNODEParams(;
    type = "SteadyStateNODE",
    initializer_layer_type = "dense",
    initializer_n_layer = 0,
    initializer_width_layers = 4,
    initializer_activation = "hardtanh",
    dynamic_layer_type = "dense",
    dynamic_hidden_states = 5,
    dynamic_n_layer = 1,
    dynamic_width_layers = 4,
    dynamic_activation = "hardtanh",
    dynamic_σ2_initialization = 0.0,
)
    SteadyStateNODEParams(
        type,
        initializer_layer_type,
        initializer_n_layer,
        initializer_width_layers,
        initializer_activation,
        dynamic_layer_type,
        dynamic_hidden_states,
        dynamic_n_layer,
        dynamic_width_layers,
        dynamic_activation,
        dynamic_σ2_initialization,
    )
end

struct SteadyStateNODEObsParams <: SurrogateModelParams
    type::String
    initializer_layer_type::String
    initializer_n_layer::Int64
    initializer_width_layers::Int64
    initializer_activation::String
    dynamic_layer_type::String
    dynamic_hidden_states::Int64
    dynamic_n_layer::Int64
    dynamic_width_layers::Int64
    dynamic_activation::String
    dynamic_σ2_initialization::Float64
    observation_layer_type::String
    observation_n_layer::Int64
    observation_width_layers::Int64
    observation_activation::String
end

function SteadyStateNODEObsParams(;
    type = "SteadyStateNODEObs",
    initializer_layer_type = "dense",
    initializer_n_layer = 0,
    initializer_width_layers = 4,
    initializer_activation = "hardtanh",
    dynamic_layer_type = "dense",
    dynamic_hidden_states = 5,
    dynamic_n_layer = 1,
    dynamic_width_layers = 4,
    dynamic_activation = "hardtanh",
    dynamic_σ2_initialization = 0.0,
    observation_layer_type = "dense",
    observation_n_layer = 0,
    observation_width_layers = 4,
    observation_activation = "hardtanh",
)
    SteadyStateNODEObsParams(
        type,
        initializer_layer_type,
        initializer_n_layer,
        initializer_width_layers,
        initializer_activation,
        dynamic_layer_type,
        dynamic_hidden_states,
        dynamic_n_layer,
        dynamic_width_layers,
        dynamic_activation,
        dynamic_σ2_initialization,
        observation_layer_type,
        observation_n_layer,
        observation_width_layers,
        observation_activation,
    )
end

struct ClassicGenParams <: SurrogateModelParams
    type::String
end

function ClassicGenParams(; type = "ClassicGenParams")
    ClassicGenParams(type)
end
