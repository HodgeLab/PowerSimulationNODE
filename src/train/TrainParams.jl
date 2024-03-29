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
- `model_params::Union{PSIDS.SteadyStateNODEParams, PSIDS.SteadyStateNODEObsParams, PSIDS.ClassicGenParams, PSIDS.GFLParams, PSIDS.GFMParams, PSIDS.ZIPParams, PSIDS.MultiDeviceParams}`: The type of surrogate model to train. Could be data-driven, physics-based, or a combination.
- `optimizer::Vector{NamedTuple{(:auto_sensealg, :algorithm, :log_η, :adjust, :initial_stepnorm, :maxiters, :steadystate_solver, :dynamic_solver, :lb_loss, :curriculum, :curriculum_timespans, :fix_params, :loss_function)}}`: Details of the optimization
    - `auto_sensealg::String`: Valid options: `"Zygote"` or `"ForwardDiff"` 
    - `algorithm::String`: Valid options: `"Adam"`, `"Bfgs"`, `"LBfgs"`. Typical choice is to start with Adam and then use BFGS. 
    - `log_η::Float64`: Log of Adam step size (ignored for other algorithms) 
    - `initial_stepnorm::Float64`: Bfgs initial step norm (ignored for other algorithms)
    - `maxiters::Int64`: Maximum number of  training iterations. 
    - `steadystate_solver::NamedTuple{(:solver, :reltol, :abstol, :termination)},`: Solver for finding initial conditions.
        - `solver::String`: the solver name from DifferentialEquations.jl
        - `reltol::Float64`: relative tolerance of the solve AND used for determining termination. 
        - `abstol::Float64`: absolute tolerance of the solve AND used for determining termination.
        - `termination::String`: how to determine when to terminate the solve. Options include `"RelSafe"` and `"RelSafeBest"`
    - `dynamic_solver::NamedTuple{(:solver, :reltol, :abstol, :maxiters, :force_tstops)}`:  Solver for dynamic trajectories.
        - `solver::String`: the solver name from DifferentialEquations.jl
        - `sensealg::String`: sensitivity algorithm for backpropogation through the dynamic solve.
        - `reltol::Float64`: relative tolearnce of the solve. 
        - `abstol::Float64`: absolute tolearnce of the solve. 
    - `maxiters::Int64`: maximum iterations of the solve. 
    - `force_tstops::Bool`: if `true`, force the solver to stop at tstops from the train dataset. If `false`, do not explicitly force any steps. 
    - `lb_loss::Float64`:  If the value of the loss on the validation set moves below lb_loss during training, the current optimization ends (current range).
    - `curriculum::String`: `"simultaneous"`: train on all of the data for each iteration of the optimizer.  `"individual faults"` cycle through the train dataset faults with one fault per iteration.  `"individual faults x2"` will run two distinct solves with the same dataset (restart the optimizer half way through) 
    - `curriculum_timespans::Vector{NamedTuple{(:tspan, :batching_sample_factor)}`: If more than one entry in `curriculum_timespans`, each fault from the input data is paired with each value of `curriculum_timespans`
        - `tspan::Tuple{Float64, Float64}`: timespan for a batch. 
        - `batching_sample_factor::Float64`: batching factor for a batch.  
    - `fix_params::Vector{Symbols}`: Valid options: `:initializer`, `:observation` (for data driven surrogates). `[]` will train all parameters for a given model. 
    - `loss_function::NamedTuple{(:α, :β, :residual_penalty)}`:
        - `α::Float64`: scales...
        - `β::Float64`: scales...
        - `residual_penalty::Float64`: scales the loss function if the implicit layer does not converge (if set to Inf, the loss is infinite anytime the implicit layer does not converge). 
- `p_start::AbstractArray`: Starting parameters (for initializer, dynamics, and observation together). By default is empty which starts with randomly initialized parameters (see `rng_seed`). 
- `check_validation_loss_iterations::Vector{Int64}`: Iterations to check the validation loss. If validation loss increases from previous iteration, the training terminates. 
- `validation_loss_termination::String`: Determine if the training should be stopped based on some observed trend in the validation loss. Set to `"false"` to never terminate. Other options: `"single increase"`: terminate training if average error on validation loss increases between iterations where it is checked (see `check_validation_loss_iterations`)
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
    model_params::PSIDS.SurrogateModelParams
    optimizer::Vector{
        NamedTuple{
            (
                :auto_sensealg,
                :algorithm,
                :log_η,
                :initial_stepnorm,
                :maxiters,
                :steadystate_solver,
                :dynamic_solver,
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
                NamedTuple{
                    (:solver, :reltol, :abstol, :termination),
                    Tuple{String, Float64, Float64, String},
                },
                NamedTuple{
                    (:solver, :sensealg, :reltol, :abstol, :maxiters, :force_tstops),
                    Tuple{String, String, Float64, Float64, Int, Bool},
                },
                Float64,
                String,
                Vector{
                    NamedTuple{
                        (:tspan, :batching_sample_factor),
                        Tuple{Tuple{Float64, Float64}, Float64},
                    },
                },
                Vector{Symbol},
                NamedTuple{(:α, :β, :residual_penalty), Tuple{Float64, Float64, Float64}},
            },
        },
    }
    p_start::AbstractArray
    check_validation_loss_iterations::Vector{Int64}
    final_validation_loss::Bool
    time_limit_buffer_seconds::Int64
    validation_loss_termination::String #"false", "single increase"
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
StructTypes.StructType(::Type{PSIDS.LineTrip}) = StructTypes.Struct()

StructTypes.StructType(::Type{PSIDS.SurrogatePerturbation}) = StructTypes.AbstractType()
StructTypes.StructType(::Type{PSIDS.SurrogateOperatingPoint}) = StructTypes.AbstractType()
StructTypes.StructType(::Type{PSIDS.SurrogateModelParams}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{PSIDS.SurrogatePerturbation}) = :type
StructTypes.subtypekey(::Type{PSIDS.SurrogateOperatingPoint}) = :type
StructTypes.subtypekey(::Type{PSIDS.SurrogateModelParams}) = :type
StructTypes.subtypes(::Type{PSIDS.SurrogatePerturbation}) = (
    PVS = PSIDS.PVS,
    Chirp = PSIDS.Chirp,
    VStep = PSIDS.VStep,
    RandomBranchTrip = PSIDS.RandomBranchTrip,
    RandomLoadTrip = PSIDS.RandomLoadTrip,
    RandomLoadChange = PSIDS.RandomLoadChange,
    LineTrip = PSIDS.LineTrip,
)
StructTypes.subtypes(::Type{PSIDS.SurrogateOperatingPoint}) = (
    GenerationLoadScale = PSIDS.GenerationLoadScale,
    ScaleSource = PSIDS.ScaleSource,
    RandomOperatingPointXiao = PSIDS.RandomOperatingPointXiao,
)

StructTypes.subtypes(::Type{PSIDS.SurrogateModelParams}) = (
    SteadyStateNODEParams = PSIDS.SteadyStateNODEParams,
    SteadyStateNODEObsParams = PSIDS.SteadyStateNODEObsParams,
    ClassicGenParams = PSIDS.ClassicGenParams,
    GFLParams = PSIDS.GFLParams,
    GFMParams = PSIDS.GFMParams,
    ZIPParams = PSIDS.ZIPParams,
    MultiDeviceParams = PSIDS.MultiDeviceParams,
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
    model_params = PSIDS.SteadyStateNODEObsParams(),
    optimizer = [
        (
            auto_sensealg = "Zygote",
            algorithm = "Adam",
            log_η = -6.0,
            initial_stepnorm = 0.01,
            maxiters = 15,
            steadystate_solver = (
                solver = "NLSolveJL",
                reltol = 1e-4,
                abstol = 1e-4,
                termination = "RelSafeBest",
            ),
            dynamic_solver = (
                solver = "Rodas5",
                sensealg = "QuadratureAdjoint",
                reltol = 1e-6,
                abstol = 1e-6,
                maxiters = 1000,
                force_tstops = true,
            ),
            lb_loss = 0.0,
            curriculum = "individual faults",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [],
            loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
        ),
    ],
    p_start = Float32[],
    check_validation_loss_iterations = [],
    final_validation_loss = true,
    time_limit_buffer_seconds = 5400,
    validation_loss_termination = "false",
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
        string("modified_validation_system_", train_id, ".json"),
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
        optimizer,
        p_start,
        check_validation_loss_iterations,
        final_validation_loss,
        time_limit_buffer_seconds,
        validation_loss_termination,
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
