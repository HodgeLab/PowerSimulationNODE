
function generate_and_train_test(p, path)
    #Generate train and validation systems
    sys_full = node_load_system(p.system_path)
    PSY.solve_powerflow!(sys_full)
    for b in PSY.get_components(PSY.Branch, sys_full)
        @warn PSY.get_name(b)
    end

    sys_train, connecting_branches =
        PSIDS.create_subsystem_from_buses(sys_full, p.surrogate_buses)
    non_surrogate_buses =
        get_number.(get_components(Bus, sys_full, x -> get_number(x) ∉ p.surrogate_buses))
    sys_validation, _ = PSIDS.create_subsystem_from_buses(sys_full, non_surrogate_buses)

    PSY.to_json(
        sys_validation,
        joinpath(
            path,
            PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
            "validation_system.json",
        ),
        force = true,
    )
    display(sys_validation)

    #Generate train, validation, and test, datasets 
    if p.train_data.system == "reduced"
        train_data = PSIDS.generate_surrogate_data(
            sys_train,   #sys_main
            sys_validation,   #sys_aux
            p.train_data.perturbations,
            p.train_data.operating_points,
            SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
            p.train_data.params,
        )
    elseif p.train_data.system == "full"
        train_data = PSIDS.generate_surrogate_data(
            sys_full,   #sys_main
            sys_validation,   #sys_aux
            p.train_data.perturbations,
            p.train_data.operating_points,
            SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
            p.train_data.params,
        )
    else
        @error "invalid parameter for the system to generate train data (should be reduced or full)"
    end
    validation_data = PSIDS.generate_surrogate_data(
        sys_full,   #sys_main
        sys_validation,  #sys_aux
        p.validation_data.perturbations,
        p.validation_data.operating_points,
        SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        p.validation_data.params,
    )
    test_data = PSIDS.generate_surrogate_data(
        sys_full,   #sys_main
        sys_validation,  #sys_aux
        p.test_data.perturbations,
        p.test_data.operating_points,
        SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        p.test_data.params,
    )

    Serialization.serialize(
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "train_data"),
        train_data,
    )
    Serialization.serialize(
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "validation_data"),
        validation_data,
    )
    Serialization.serialize(
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "test_data"),
        test_data,
    )

    display(p)
    status, θ = train(p, connecting_branches)    #TODO -rework training to incorporate new data files, parameters, etc. 
    @test status
    input_param_file = joinpath(path, "input_data", "input_test1.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    visualize_training(input_param_file, skip = 1)
    animate_training(input_param_file, skip = 1)
    a = generate_summary(joinpath(path, "output_data"))
    p = visualize_summary(a)
    #display(Plots.plot(p))
    print_high_level_output_overview(a, path)
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
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas4",
                formulation = "MassMatrix",
                solver_tols = (1e-4, 1e-4),
            ),
            system = "full",     #generate from the reduced system with sources to perturb or the full system
        ),
        validation_data = (
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.RandomLoadChange(time = 1.0, load_multiplier_range = (0.9, 1.1)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas4",
                formulation = "MassMatrix",
                solver_tols = (1e-2, 1e-2),
            ),
        ),
        test_data = (
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale(
                generation_scale = 1.0,
                load_scale = 1.0,
            )],
            perturbations = [[
                PSIDS.RandomLoadChange(time = 0.5, load_multiplier_range = (0.9, 1.1)),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas4",
                formulation = "MassMatrix",
                solver_tols = (1e-4, 1e-4),
            ),
        ),
        validation_loss_every_n = 200,
        steady_state_solver = (
            solver = "SSRootfind", #"SSRootfind",
            abstol = 1e-8,       #xtol, ftol  #High tolerance -> standard NODE with initializer and observation 
            maxiters = 1e3,   #don't think this has any impact - not implemented correctly 
        ),
        maxiters = 5,
        model_initializer = (
            type = "dense",     #OutputParams (train initial conditions)
            n_layer = 0,
            width_layers = 4,
            activation = "hardtanh",
        ),
        model_node = (
            type = "dense",
            n_layer = 1,
            width_layers = 4,
            activation = "relu",
            initialization = "default",
        ),
        model_observation = (
            type = "dense",
            n_layer = 0,
            width_layers = 4,
            activation = "relu",
        ),
        optimizer = (
            sensealg = "Zygote",
            primary = "Adam",
            primary_η = 0.001,
            adjust = "nothing",
            adjust_η = 0.0,
        ),
        input_normalization = (
            x_scale = [1.0, 1.0, 1.0, 1.0],  #TODO - set defaults 
            x_bias = [0.0, 0.0, 0.0, 0.0],
            exogenous_scale = [20.0, 20.0],
            exogenous_bias = [-1.0, -1.0],
        ),
    )
    try
        generate_and_train_test(p, path)
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

#TODO - test train visualization functionality
#TODO - test parameter restart functionality 
