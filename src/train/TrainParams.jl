"""
    mutable struct TrainParams  

# Fields
- `train_id::String`: id for the training instance, used for naming output data folder.
- `surrogate_buses::Vector{Int64}`: The numbers of the buses which make up the portion of the system to be replaced with a surrogate.
- `train_data::NamedTuple{(:id, :operating_points, :perturbations, :params, :system)}`: 
    - `id::String`: For identifying repeated train data sets for hyper parameter tuning. 
    - `operating_points::Vector{PSIDS.SurrogateOperatingPoint}`: 
    - `perturbations::Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}}`:  
    - `params::PSIDS.GenerateDataParams`: 
    - `system::String`: options are `"full"` and `"reduced"`.  
- `validation_data::NamedTuple{(:id, :operating_points, :perturbations, :params)}`: 
    - `id::String`: For identifying repeated train data sets for hyper parameter tuning. 
    - `operating_points::Vector{PSIDS.SurrogateOperatingPoint}`:
    - `perturbations::Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}}`:  
    - `params::PSIDS.GenerateDataParams`: 
- `validation_data::NamedTuple{(:id, :operating_points, :perturbations, :params)}`: 
    - `id::String`: For identifying repeated train data sets for hyper parameter tuning. 
    - `operating_points::Vector{PSIDS.SurrogateOperatingPoint}`:  
    - `perturbations::Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}}`:  
    - `params::PSIDS.GenerateDataParams`: 
- `model_params::Union{SteadyStateNODEParams, SteadyStateNODEObsParams, ClassicGenParams}`: The type of surrogate model to train. Could be data-driven, physics-based, or a combination.
- `steady_state_solver::NamedTuple{(:solver, :abstol, :maxiters)}`: Solver for finding initial conditions.
    - `solver::String`: the solver name from DifferentialEquations.jl
    - `abstol::Float64`: absolute tolerance of the solve.
    - `maxiters::Int64`: maximum iterations of the solve. 
- `dynamic_solver::NamedTuple{(:solver, :reltol, :abstol, :maxiters, :force_tstops)}`:  Solver for dynamic trajectories.
    - `solver::String`: the solver name from DifferentialEquations.jl
    - `reltol::Float64`: relative tolearnce of the solve. 
    - `abstol::Float64`: absolute tolearnce of the solve. 
    - `maxiters::Int64`: maximum iterations of the solve. 
    - `force_tstops::Bool`: if `true`, force the solver to stop at tstops from the train dataset. If `false`, do not explicitly force any steps. 
- `optimizer::Vector{NamedTuple{(:sensealg, :algorithm, :η, :adjust, :initial_stepnorm, :maxiters, :lb_loss, :curriculum, :curriculum_timespans, :fix_params, :loss_function)}}`: Details of the optimization
    - `sensealg::String`: Valid options: `"Zygote"` or `"ForwardDiff"` 
    - `algorithm::String`: Valid options: `"Adam"`, `"Bfgs"`, `"LBfgs"`. Typical choice is to start with Adam and then use BFGS. 
    - `η::Float64`: Adam step size (ignored for other algorithms) 
    - `initial_stepnorm::Float64`: Bfgs initial step norm (ignored for other algorithms)
    - `maxiters::Int64`: Maximum number of  training iterations. 
    - `lb_loss::Float64`:  If the value of the loss on the validation set moves below lb_loss during training, the current optimization ends (current range).
    - `curriculum::String`: `"simultaneous"`: train on all of the data for each iteration of the optimizer.  `"individual faults"` cycle through the train dataset faults with one fault per iteration.  `"individual faults x2"` will run two distinct solves with the same dataset (restart the optimizer half way through) 
    - `curriculum_timespans::Vector{NamedTuple{(:tspan, :batching_sample_factor)}`: If more than one entry in `curriculum_timespans`, each fault from the input data is paired with each value of `curriculum_timespans`
        - `tspan::Tuple{Float64, Float64}`: timespan for a batch. 
        - `batching_sample_factor::Float64`: batching factor for a batch.  
    - `fix_params::Vector{Symbols}`: Valid options: `:initializer`, `:observation` (for data driven surrogates). `[]` will train all parameters for a given model. 
    - `loss_function::NamedTuple{(:component_weights, :type_weights)}`:
        - `component_weights::NamedTuple{(:initialization_weight, :dynamic_weight, :residual_penalty)}`
            - `initialization_weight::Float64`: scales the portion of loss function penalizing the difference between predicted steady state and actual. 
            - `dynamic_weight::Float64`: scales the portion of the loss function penalizing differences in the output time series.
            - `residual_penalty::Float64`: scales the loss function if the implicit layer does not converge (if set to Inf, the loss is infinite anytime the implicit layer does not converge). 
        - `type_weights::NamedTuple{(:rmse, :mae)}`: (sum to one)
            - `rmse::Float64`: weight for root mean square error loss formulation.
            - `mae::Float64`: weight for mean absolute error loss formulation.
- `p_start::Vector{Float32}`: Starting parameters (for initializer, dynamics, and observation together). By default is empty which starts with randomly initialized parameters (see `rng_seed`). 
- `validation_loss_every_n::Int64`: Determines how often, during training, the surrogate is added to the full system and loss is evaluated. 
- `rng_seed::Int64`: Seed for the random number generator used for initializing the NN for reproducibility across training runs.
- `output_mode_skip::Int`: Record and save output data every `output_mode_skip` iterations. Meant to ease memory constraints on HPC. 
- `train_time_limit_seconds::Int64`:  
- `base_path:String`: Directory for training where input data is found and output data is written.
- `system_path::String`: Location of the full `System`. Training/validation/test systems are dervied based on `surrogate_buses`.  
- `surrogate_system_path`: Path to validation system (before surrogate is added to this system during training).
- `modified_surrogate_system_path`:  Path to validation system (after surrogate is added to this system during training). 
- `train_system_path`: Path to train system (system with only the surrogate represented in detail with sources surrounding).
- `train_data_path::String`: path to train data. 
- `validation_data_path::String`: path to validation data.
- `test_data_path::String`: path to test_data. 
- `output_data_path:String`: From `base_path`, the directory for saving output data.
"""
mutable struct TrainParams
    train_id::String
    surrogate_buses::Vector{Int64}
    train_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params, :system),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
            String,
        },
    }
    validation_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
        },
    }
    test_data::NamedTuple{
        (:id, :operating_points, :perturbations, :params),
        Tuple{
            String,
            Vector{PSIDS.SurrogateOperatingPoint},
            Vector{Vector{Union{PSIDS.SurrogatePerturbation, PSID.Perturbation}}},
            PSIDS.GenerateDataParams,
        },
    }
    model_params::SurrogateModelParams
    steady_state_solver::NamedTuple{
        (:solver, :abstol, :maxiters),
        Tuple{String, Float64, Int},
    }
    dynamic_solver::NamedTuple{
        (:solver, :reltol, :abstol, :maxiters, :force_tstops),
        Tuple{String, Float64, Float64, Int, Bool},
    }
    optimizer::Vector{
        NamedTuple{
            (
                :sensealg,
                :algorithm,
                :η,
                :initial_stepnorm,
                :maxiters,
                :lb_loss,
                :curriculum,
                :curriculum_timespans,
                :fix_params,
                :loss_function,
            ),
            Tuple{
                String,
                String,
                Float64,
                Float64,
                Int64,
                Float64,
                String,
                Vector{
                    NamedTuple{
                        (:tspan, :batching_sample_factor),
                        Tuple{Tuple{Float64, Float64}, Float64},
                    },
                },
                Vector{Symbol},
                NamedTuple{
                    (:component_weights, :type_weights),
                    Tuple{
                        NamedTuple{
                            (:initialization_weight, :dynamic_weight, :residual_penalty),
                            Tuple{Float64, Float64, Float64},
                        },
                        NamedTuple{(:rmse, :mae), Tuple{Float64, Float64}},
                    },
                },
            },
        },
    }
    p_start::Vector{Float32}
    validation_loss_every_n::Int64
    rng_seed::Int64
    output_mode_skip::Int64
    train_time_limit_seconds::Int64
    base_path::String
    system_path::String
    surrogate_system_path::String
    modified_surrogate_system_path::String
    train_system_path::String
    data_collection_location_path::String
    train_data_path::String
    validation_data_path::String
    test_data_path::String
    output_data_path::String
end

StructTypes.StructType(::Type{TrainParams}) = StructTypes.Mutable()
StructTypes.StructType(::Type{PSIDS.GenerateDataParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.PVS}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.Chirp}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.VStep}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.GenerationLoadScale}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.ScaleSource}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomBranchTrip}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomLoadTrip}) = StructTypes.Struct()
StructTypes.StructType(::Type{PSIDS.RandomLoadChange}) = StructTypes.Struct()

StructTypes.StructType(::Type{PSIDS.SurrogatePerturbation}) = StructTypes.AbstractType()
StructTypes.StructType(::Type{PSIDS.SurrogateOperatingPoint}) = StructTypes.AbstractType()
StructTypes.StructType(::Type{SurrogateModelParams}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{PSIDS.SurrogatePerturbation}) = :type
StructTypes.subtypekey(::Type{PSIDS.SurrogateOperatingPoint}) = :type
StructTypes.subtypekey(::Type{SurrogateModelParams}) = :type
StructTypes.subtypes(::Type{PSIDS.SurrogatePerturbation}) = (
    PVS = PSIDS.PVS,
    Chirp = PSIDS.Chirp,
    VStep = PSIDS.VStep,
    RandomBranchTrip = PSIDS.RandomBranchTrip,
    RandomLoadTrip = PSIDS.RandomLoadTrip,
    RandomLoadChange = PSIDS.RandomLoadChange,
)
StructTypes.subtypes(::Type{PSIDS.SurrogateOperatingPoint}) =
    (GenerationLoadScale = PSIDS.GenerationLoadScale, ScaleSource = PSIDS.ScaleSource)

StructTypes.subtypes(::Type{SurrogateModelParams}) = (
    SteadyStateNODEParams = SteadyStateNODEParams,
    SteadyStateNODEObsParams = SteadyStateNODEObsParams,
    ClassicGenParams = ClassicGenParams,
)

function TrainParams(;
    train_id = "train_instance_1",
    surrogate_buses = [1],
    train_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.PVS(source_name = "InfBus")]],
        params = PSIDS.GenerateDataParams(),
        system = "reduced",     #generate from the reduced system with sources to perturb or the full system
    ),
    validation_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.VStep(source_name = "InfBus")]],    #To do - make this a branch impedance double 
        params = PSIDS.GenerateDataParams(),
    ),
    test_data = (
        id = "1",
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.VStep(source_name = "InfBus")]],    #To do - make this a branch impedance double 
        params = PSIDS.GenerateDataParams(),
    ),
    model_params = SteadyStateNODEObsParams(),
    steady_state_solver = (
        solver = "SSRootfind",
        abstol = 1e-4,       #xtol, ftol  #High tolerance -> standard NODE with initializer and observation 
        maxiters = 5,
    ),
    dynamic_solver = (
        solver = "Rodas5",
        reltol = 1e-6,
        abstol = 1e-6,
        maxiters = 1000,
        force_tstops = true,
    ),
    optimizer = [
        (
            sensealg = "Zygote",
            algorithm = "Adam",
            η = 0.000001,
            initial_stepnorm = 0.01,
            maxiters = 15,
            lb_loss = 0.0,
            curriculum = "individual faults",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [],
            loss_function = (
                component_weights = (
                    initialization_weight = 1.0,
                    dynamic_weight = 1.0,
                    residual_penalty = 1.0e9,
                ),
                type_weights = (rmse = 1.0, mae = 0.0),
            ),
        ),
    ],
    p_start = Float32[],
    validation_loss_every_n = 100,
    rng_seed = 123,
    output_mode_skip = 1,
    train_time_limit_seconds = 1e9,
    base_path = pwd(),
    system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "system.json",
    ),
    surrogate_system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "validation_system.json",
    ),
    modified_surrogate_system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "modified_validation_system.json",
    ),
    train_system_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "train_system.json",
    ),
    data_collection_location_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        "connecting_branches_names",
    ),
    train_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "train_data_$(train_data.id)",
    ),
    validation_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "validation_data_$(validation_data.id)",
    ),
    test_data_path = joinpath(
        base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        "test_data_$(test_data.id)",
    ),
    output_data_path = joinpath(base_path, "output_data"),
)
    TrainParams(
        train_id,
        surrogate_buses,
        train_data,
        validation_data,
        test_data,
        model_params,
        steady_state_solver,
        dynamic_solver,
        optimizer,
        p_start,
        validation_loss_every_n,
        rng_seed,
        output_mode_skip,
        train_time_limit_seconds,
        base_path,  #other paths derived from this one must come after 
        system_path,
        surrogate_system_path,
        modified_surrogate_system_path,
        train_system_path,
        data_collection_location_path,
        train_data_path,
        validation_data_path,
        test_data_path,
        output_data_path,
    )
end

function TrainParams(file::AbstractString)
    return JSON3.read(read(file), TrainParams)
end

"""
    serialize(inputs::TrainParams, file_path::String)

Serializes  the input to JSON file.
"""
function serialize(inputs::TrainParams, file_path::String)
    open(file_path, "w") do io
        JSON3.write(io, inputs)
    end
    return
end

function Base.show(io::IO, ::MIME"text/plain", params::TrainParams)
    for field_name in fieldnames(TrainParams)
        field = getfield(params, field_name)
        if isa(field, NamedTuple)
            println(io, "$field_name =")
            for k in keys(field)
                println(io, "\t", k, " : ", field[k])
            end
        else
            println(io, "$field_name = ", getfield(params, field_name))
        end
    end
end
