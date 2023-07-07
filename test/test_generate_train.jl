TEST_ATOL = 1e-5
function generate_and_train_test(p2)
    input_param_file = joinpath(p2.base_path, "input_data", "input_test1.json")
    PowerSimulationNODE.serialize(p2, input_param_file)
    p = TrainParams(input_param_file)
    build_subsystems(p)
    generate_train_data(p)
    generate_validation_data(p)
    generate_test_data(p)

    status, θ = train(p)
    @test status
    input_param_file = joinpath(p.base_path, "input_data", "input_test1.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    #visualize_training(input_param_file, [1, 2, 3]) #BUG: running this function the arrow files are not closed properly and you can't delete the directory with rm()
    #animate_training(input_param_file, [1, 2, 3]) - internal bug? 
    a = generate_summary(joinpath(p.base_path, "output_data"))
    pp = visualize_summary(a)
    print_high_level_output_overview(a, p.base_path)

    #Plot real and imag current for a full dataset
    data_collection_location_validation =
        Serialization.deserialize(p.data_collection_location_path)[2]
    surrogate_dataset = generate_surrogate_dataset(
        System(p.modified_surrogate_system_path),
        System(p.surrogate_system_path),
        θ,
        Serialization.deserialize(p.validation_data_path),
        p.validation_data,
        data_collection_location_validation,
        p.model_params,
    )
    ps =
        visualize_loss(surrogate_dataset, Serialization.deserialize(p.validation_data_path))

    #Evaluate loss metrics for a full dataset 

    surrogate_dataset = generate_surrogate_dataset(
        System(p.modified_surrogate_system_path),
        System(p.surrogate_system_path),
        θ,
        Serialization.deserialize(p.validation_data_path),
        p.validation_data,
        Serialization.deserialize(p.data_collection_location_path)[2],
        p.model_params,
    )
    dataset_loss =
        evaluate_loss(surrogate_dataset, Serialization.deserialize(p.validation_data_path))
    return dataset_loss
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

@testset "SteadyStateNODEObs (9 bus system, train-data from full system)" begin
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
        model_params = PSIDS.SteadyStateNODEObsParams(
            name = "source_1",
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers_relative_input = 1,
            initializer_activation = "tanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -5,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
            observation_layer_type = "dense",
            observation_n_layer = 0,
            observation_width_layers_relative_input = -1,
            observation_activation = "relu",
        ),
        optimizer = [(
            auto_sensealg = "Zygote",
            algorithm = "Adam",
            log_η = -10.0,
            initial_stepnorm = 0.0, #ignored for ADAM 
            maxiters = 6,
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
                maxiters = 1e5,
                force_tstops = true,
            ),
            lb_loss = 0.0,
            curriculum = "simultaneous",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [:initializer, :observation],
            loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
        )],
        p_start = [],
        check_validation_loss_iterations = [1, 2, 3, 4, 5, 6],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.007453259120865807,
                    atol = TEST_ATOL,
                ) =#
        #loss_dataframe =  PowerSimulationNODE.read_arrow_file_to_dataframe(joinpath(p.output_data_path,"train_instance_1", "loss"))
        #@test loss_dataframe[6, :iteration_time_seconds] - loss_dataframe[5, :iteration_time_seconds] < 1.0
        #GC.gc()
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "SteadyStateNODE (9 bus system, train-data from full system, input starting parameters)" begin
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
    include(joinpath(TEST_FILES_DIR, "scripts", "build_9bus.jl"))
    SURROGATE_BUSES = [2]
    branch_to_trip = "4-6-i_5"
    path, full_system_path = _generic_test_setup()
    Random.seed!(1) #starting parameters are generated randomly--> need to make sure values are consistent. 
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
        model_params = PSIDS.SteadyStateNODEParams(
            name = "source_1",
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers_relative_input = 1,
            initializer_activation = "tanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -5,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
        ),
        optimizer = [(
            auto_sensealg = "Zygote",
            algorithm = "Adam",
            log_η = -10.0,
            initial_stepnorm = 0.0, #ignored for ADAM 
            maxiters = 6,
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
                maxiters = 1e5,
                force_tstops = true,
            ),
            lb_loss = 0.0,
            curriculum = "individual faults",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [],
            loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
        )],
        p_start = Float32.(rand(88)), #will be completely unstable, making sure training still runs.
        check_validation_loss_iterations = [20],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(dataset_loss["max_error_ir"][1], 0.004614147683377423, atol = TEST_ATOL) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "SteadyStateNODE (9 bus system, train-data from full system)" begin
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
        model_params = PSIDS.SteadyStateNODEParams(
            name = "source_1",
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers_relative_input = 1,
            initializer_activation = "tanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -5,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
        ),
        optimizer = [(
            auto_sensealg = "Zygote",
            algorithm = "Adam",
            log_η = -10.0,
            initial_stepnorm = 0.0, #ignored for ADAM 
            maxiters = 6,
            steadystate_solver = (
                solver = "NLSolveJL",
                reltol = 1e-4,
                abstol = 1e-4,
                termination = "RelSafe",
            ),
            dynamic_solver = (
                solver = "Rodas5",
                sensealg = "QuadratureAdjoint",
                reltol = 1e-6,
                abstol = 1e-6,
                maxiters = 1e5,
                force_tstops = true,
            ),
            lb_loss = 0.0,
            curriculum = "individual faults",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [],
            loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
        )],
        p_start = [],
        check_validation_loss_iterations = [20],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.00384531814395217,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

#= @testset "SteadyStateNODE (9 bus system, train-data from full system, BFGS)" begin
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
                tsave = [],
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
        model_params = PSIDS.SteadyStateNODEParams(    #Changed from SS node params 
            name = "source_1",
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers_relative_input = 1,
            initializer_activation = "tanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -5,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
        ),
        optimizer = [(
            auto_sensealg = "Zygote",
            algorithm = "Bfgs",
            log_η = -10.0,
            initial_stepnorm = 0.0, #ignored for ADAM 
            maxiters = 6,
            steadystate_solver = (
                solver = "NLSolveJL",
                reltol = 1e-4,
                abstol = 1e-4,
                termination = "RelSafe",
            ),
            dynamic_solver = (
                solver = "Rodas5",
                sensealg = "QuadratureAdjoint",  #changed from Quadtrature 
                reltol = 1e-6,
                abstol = 1e-6,
                maxiters = 1e5,
                force_tstops = true,
            ),
            lb_loss = 0.0,
            curriculum = "individual faults",
            curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
            fix_params = [],
            loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
        )],
        p_start = [],
        check_validation_loss_iterations = [20],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.0037364045897945175,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end =#

@testset "SteadyStateNODEObs (9 bus system, train-data from reduced system)" begin
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
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.ScaleSource(
                source_name = "source_1",
                V_scale = 1.0,
                θ_scale = 1.0,
                P_scale = 1.0,
                Q_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.Chirp(;
                    source_name = "source_1",  #when building training system, sources are named source_$(ix) for each port
                    ω1 = 1.0,
                    ω2 = 2.0,
                    tstart = 0.5,
                    N = 1.0,
                    V_amp = 0.1,
                    ω_amp = 0.1,
                ),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
                tspan = (0.0, 1.0),
                tstops = [0.0, 0.5, 1.0], #issue with tstop at 0.0 with dynamic lines? 
                tsave = [],
                all_lines_dynamic = false,
                all_branches_dynamic = false,   #Can't do dynamic transformers? 
                seed = 1,
            ),
            system = "reduced",
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
        model_params = PSIDS.SteadyStateNODEObsParams(
            name = "source_1",
            initializer_layer_type = "dense",
            initializer_n_layer = 0,
            initializer_width_layers_relative_input = 1,
            initializer_activation = "tanh",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -5,
            dynamic_activation = "relu",
            dynamic_σ2_initialization = 0.0,
            observation_layer_type = "dense",
            observation_n_layer = 0,
            observation_width_layers_relative_input = -1,
            observation_activation = "relu",
        ),
        optimizer = [
            (
                auto_sensealg = "Zygote",
                algorithm = "Adam",
                log_η = -10.0,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
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
                    maxiters = 1e5,
                    force_tstops = true,
                ),
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = [:initializer, :observation],
                loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
            ),
        ],
        p_start = [],
        check_validation_loss_iterations = [],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.7287053003959845,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "ClassicGen (9 bus system, train-data from full system)" begin
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
        model_params = PSIDS.ClassicGenParams(name = "source_1"),
        optimizer = [
            (
                auto_sensealg = "ForwardDiff",
                algorithm = "Adam",
                log_η = -1.0,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
                steadystate_solver = (
                    solver = "NLSolveJL",
                    reltol = 1e-4,
                    abstol = 1e-4,
                    termination = "RelSafeBest",
                ),
                dynamic_solver = (
                    solver = "Rodas5",
                    sensealg = "ForwardDiffSensitivity",
                    reltol = 1e-6,
                    abstol = 1e-6,
                    maxiters = 1e5,
                    force_tstops = true,
                ),
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = [],
                loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
            ),
        ],
        p_start = [],
        check_validation_loss_iterations = [],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.0012409126130332737,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "GFL (9 bus system, train-data from full system)" begin
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
        model_params = PSIDS.GFLParams(name = "source_1"),
        optimizer = [
            (
                auto_sensealg = "ForwardDiff",
                algorithm = "Adam",
                log_η = -4.0,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
                steadystate_solver = (
                    solver = "NLSolveJL",
                    reltol = 1e-4,
                    abstol = 1e-4,
                    termination = "RelSafeBest",
                ),
                dynamic_solver = (
                    solver = "Rodas5",
                    sensealg = "ForwardDiffSensitivity",
                    reltol = 1e-6,
                    abstol = 1e-6,
                    maxiters = 1e5,
                    force_tstops = true,
                ),
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = [],
                loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
            ),
        ],
        p_start = Float64[],
        check_validation_loss_iterations = [],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        @test dataset_loss["max_error_ir"] == [0.0]
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "GFM (9 bus system, train-data from full system)" begin
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
                [PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.99, 1.01))],
            ],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
                tspan = (0.0, 1.0),
                tstops = [0.0, 0.5, 1.0], #[0.0, 0.5, 1.0],  #issue with tstop at 0.0 with dynamic lines? 
                tsave = [0.0, 0.5, 1.0], #[0.0,0.5,1.0],#0:0.01:1.0,# [], # 0:0.01:1.0,
                all_lines_dynamic = true,
                all_branches_dynamic = true,   #Can't do dynamic transformers? 
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
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (1.0, 1.0)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
            ),
        ),
        model_params = PSIDS.GFMParams(name = "source_1"),
        optimizer = [
            (
                auto_sensealg = "ForwardDiff",
                algorithm = "Adam",
                log_η = -10.0,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
                steadystate_solver = (
                    solver = "NLSolveJL",
                    reltol = 1e-4,
                    abstol = 1e-4,
                    termination = "RelSafeBest",
                ),
                dynamic_solver = (
                    solver = "Rodas5",
                    sensealg = "ForwardDiffSensitivity",
                    reltol = 1e-1,  #Only solves at lower tolerance. 
                    abstol = 1e-1,
                    maxiters = 1e5,
                    force_tstops = true,
                ),
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = [],
                loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
            ),
        ],
        p_start = Float64[],
        check_validation_loss_iterations = [],
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.00159187565598784,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end

@testset "MultiDevice (9 bus system, train-data from full system)" begin
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
                [PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.99, 1.01))],
            ],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
                tspan = (0.0, 1.0),
                tstops = [0.0, 0.5, 1.0], #[0.0, 0.5, 1.0],  #issue with tstop at 0.0 with dynamic lines? 
                tsave = [0.0, 0.5, 1.0], #[0.0,0.5,1.0],#0:0.01:1.0,# [], # 0:0.01:1.0,
                all_lines_dynamic = true,
                all_branches_dynamic = true,   #Can't do dynamic transformers? 
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
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (1.0, 1.0)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-4, abstol = 1e-4),
            ),
        ),
        model_params = PSIDS.MultiDeviceParams(name = "source_1"),
        optimizer = [
            (
                auto_sensealg = "ForwardDiff",
                algorithm = "Adam",
                log_η = -10.0,
                initial_stepnorm = 0.0, #ignored for ADAM 
                maxiters = 6,
                steadystate_solver = (
                    solver = "NLSolveJL",
                    reltol = 1e-4,
                    abstol = 1e-4,
                    termination = "RelSafeBest",
                ),
                dynamic_solver = (
                    solver = "Rodas5",
                    sensealg = "ForwardDiffSensitivity",
                    reltol = 1e-1,  #Only solves at lower tolerance. 
                    abstol = 1e-1,
                    maxiters = 1e5,
                    force_tstops = true,
                ),
                lb_loss = 0.0,
                curriculum = "simultaneous",
                curriculum_timespans = [(tspan = (0.0, 1.0), batching_sample_factor = 1.0)],
                fix_params = [
                    :P_fraction_1,
                    :P_fraction_2,
                    :P_fraction_3,
                    :Q_fraction_1,
                    :kffv_gfl,
                    :kffv_gfm,
                    :kffi,
                ],
                loss_function = (α = 0.5, β = 1.0, residual_penalty = 1.0e9),
            ),
        ],
        p_start = Float64[
            -0.2,
            0.4,
            0.4,
            0.0,
            100.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            100.0,
            12.700000000000001,
            615.9420289855071,
            1.9835456702225953,
            30.679163077003263,
            41.818017263110214,
            3.899221794528066,
            30.29304511089826,
            42.88608307322704,
            5.813096036884919,
            11.866710997978586,
            0.0,
            600.0,
            512.5954002113241,
            0.08625118210709415,
            4.482557140304551,
            0.07999999999999999,
            0.0029999999999999996,
            0.07399999999999998,
            0.19999999999999998,
            0.009999999999999998,
            100.0,
            12.700000000000001,
            205.3140096618357,
            0.04991162961714421,
            30.41129394073096,
            0.2018427749817273,
            1006.7417873074656,
            0.579584126157358,
            752.8680589602519,
            0.0,
            0.0,
            0.19950887723724778,
            1.2844250396816308,
            1.2844250396816308,
            0.0,
            47.58554332620469,
            0.1975149812011315,
            600.0,
            0.07999999999999999,
            0.0029999999999999996,
            0.07399999999999998,
            0.19999999999999998,
            0.009999999999999998,
        ],
        check_validation_loss_iterations = [],
        final_validation_loss = false,
        validation_loss_termination = "false",
        output_mode_skip = 1,
    )
    try
        dataset_loss = generate_and_train_test(p)
        #=         @test isapprox(
                    dataset_loss["max_error_ir"][1],
                    0.0021414167152631336,
                    atol = TEST_ATOL,
                ) =#
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end
