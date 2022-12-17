
function generate_and_train_test(p)
    build_subsystems(p)
    generate_train_data(p)
    generate_validation_data(p)
    generate_test_data(p)

    status, θ = train(p)
    @test status
    input_param_file = joinpath(p.base_path, "input_data", "input_test1.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    visualize_training(input_param_file, skip = 1)
    #animate_training(input_param_file, skip = 1)       #TODO - internal bug in animation
    a = generate_summary(joinpath(p.base_path, "output_data"))
    pp = visualize_summary(a)
    print_high_level_output_overview(a, p.base_path)

    #θ_ranges is needed to split the full parameter vector (θ) into the parts of the surrogate.
    θ_ranges = JSON3.read(
        read(joinpath(p.base_path, "output_data", p.train_id, "high_level_outputs")),
        Dict{String, Any},
    )["θ_ranges"]
    @warn θ_ranges

    #Plot real and imag current for a full dataset  (TODO- document)
    ps = visualize_loss(
        System(p.surrogate_system_path),
        θ,
        Serialization.deserialize(p.validation_data_path),
        p.validation_data,
        Serialization.deserialize(p.data_collection_location_path)[2],
        θ_ranges,
    )

    #Evaluate loss metrics for a full dataset 
    _ = evaluate_loss(
        System(p.surrogate_system_path),
        θ,
        Serialization.deserialize(p.validation_data_path),
        p.validation_data,
        Serialization.deserialize(p.data_collection_location_path)[2],
        θ_ranges,
    )
end

function _generic_test_setup()
    path = (joinpath(pwd(), "test-train-dir"))
    !isdir(path) && mkdir(path)
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME))
    mkpath(joinpath(path, "scripts"))
    full_system_path =
        joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "full_system.json")
    cp(
        joinpath(TEST_FILES_DIR, "system_data", "full_system.json"),
        full_system_path,
        force = true,
    )
    cp(
        joinpath(TEST_FILES_DIR, "system_data", "full_system_validation_descriptors.json"),
        joinpath(path, "system_data", "full_system_validation_descriptors.json"),
        force = true,
    )
    return path, full_system_path
end

@testset "9 bus system, train-data from full system" begin
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
    include(joinpath(TEST_FILES_DIR, "scripts", "build_9bus.jl"))
    SURROGATE_BUSES = [2]
    branch_to_trip = "4-6-i_5"
    path, full_system_path = _generic_test_setup()

    p = TrainParams(
        base_path = path,
        surrogate_buses = SURROGATE_BUSES,
        system_path = full_system_path,
        train_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [
                [PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1))],
                [PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1))],
            ],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
                tspan = (0.0, 1.0),
                tstops = [0.0, 0.5, 1.0], #[0.0, 0.5, 1.0],  #issue with tstop at 0.0 with dynamic lines? 
                tsave = [], #[0.0,0.5,1.0],#0:0.01:1.0,# [], # 0:0.01:1.0,
                all_lines_dynamic = false,
                all_branches_dynamic = false,   #Can't do dynamic transformers? 
                seed = 1,
            ),
            system = "full",
        ),
        validation_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
                tspan = (0.0, 1.0),
                tstops = [0.0, 0.5, 1.0], #[0.0, 0.5, 1.0],  #issue with tstop at 0.0 with dynamic lines? 
                tsave = [0.0, 0.5, 1.0],  #must have tstops with validation data 
                all_lines_dynamic = false,
                all_branches_dynamic = false,   #Can't do dynamic transformers? 
                seed = 1,
            ),
        ),
        test_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
            ),
        ),
        model_params = SteadyStateNODEObsParams(
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers = 4,
            initializer_activation = "hardtanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers = 4,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
            observation_layer_type = "dense",
            observation_n_layer = 0,
            observation_width_layers = 4,
            observation_activation = "relu",
        ),
        steady_state_solver = (
            solver = "SSRootfind",
            abstol = 1e-4,       #xtol, ftol  #High tolerance -> standard NODE with initializer and observation 
            maxiters = 1e3,   #TODO - don't think this has any impact - not implemented correctly? check  
        ),
        dynamic_solver = (
            solver = "Rodas5",
            reltol = 1e-6,
            abstol = 1e-6,
            maxiters = 1e5,
            force_tstops = true,
        ),
        optimizer = [
            (
                sensealg = "Zygote",
                algorithm = "Adam", #"Bfgs", "Adam"
                η = 0.0000000001,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = "initializer+observation",
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
        p_start = [],
        validation_loss_every_n = 20,
        output_mode_skip = 1,
    )
    try
        generate_and_train_test(p)
        #=         @test generate_summary(p.output_data_path)["train_instance_1"]["timing_stats"][1]["time"] <
                      11.0 #should pass after precompilation run  =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

#= @testset "2 bus system, train-data from reduced system" begin
include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
include(joinpath(TEST_FILES_DIR, "scripts", "build_2bus.jl"))      
SURROGATE_BUS = [102]  
branch_to_trip = "BUS 1-BUS 2-i_1"
path, full_system_path = _generic_test_setup()
p = TrainParams(
    base_path = path,
    surrogate_buses = SURROGATE_BUS,
    system_path = full_system_path,
    train_data = (
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[PSIDS.PVS(source_name = "source_1")]],
        params = PSIDS.GenerateDataParams(),
        system = "reduced",     #generate from the reduced system with sources to perturb or the full system
    ),
    validation_data = (
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[
            PSID.BranchImpedanceChange(0.5, PSY.Line, branch_to_trip , 2.0),
        ]],
        params = PSIDS.GenerateDataParams(),
    ),
    test_data = (
        operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
        perturbations = [[ PSID.BranchImpedanceChange(0.5, PSY.Line,branch_to_trip, 2.0),]],   
        params = PSIDS.GenerateDataParams(),
    ),
)
generate_and_train(p, path)
end   

@testset "9 bus system, train-data from reduced system" begin
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
    include(joinpath(TEST_FILES_DIR, "scripts", "build_9bus.jl"))      
    SURROGATE_BUSES = [2] 
    branch_to_trip = "4-6-i_5"
    path, full_system_path = _generic_test_setup()
    p = TrainParams(
        base_path = path,
        surrogate_buses = SURROGATE_BUSES,
        system_path = full_system_path,
        train_data = (
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations =  [[PSIDS.PVS(source_name = "source_1")]],
            params = PSIDS.GenerateDataParams(),
            system = "reduced",     #generate from the reduced system with sources to perturb or the full system
        ),
        validation_data = (
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations = [[
                PSID.BranchImpedanceChange(0.5, PSY.Line, branch_to_trip, 2.0),
            ]],
            params = PSIDS.GenerateDataParams(),
        ),
        test_data = (
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations = [[ PSID.BranchImpedanceChange(0.5, PSY.Line, branch_to_trip, 2.0),]],   
            params = PSIDS.GenerateDataParams(),
        ),
    )
    generate_and_train_test(p, path)
end  =#
#TODO add different parameter cases 
#TODO - test parameter restart functionality 
