include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))

include(joinpath(TEST_FILES_DIR, "scripts", "build_14bus.jl"))      #Change which system you want to use 
SURROGATE_BUS = 16 #SURROGATE_BUS = 102  
fault_generator = "generator-15-1" #  "generator-102-1"
train_from_coefficients = true
path = (joinpath(pwd(), "test-train-dir"))
!isdir(path) && mkdir(path)

try
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME))
    mkpath(joinpath(path, "scripts"))

    full_system_path =
        joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "full_system.json")
    yaml_path = joinpath(path, "scripts", "config.yml")

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
    cp(joinpath(TEST_FILES_DIR, "scripts", "config.yml"), yaml_path, force = true)

    sys_full = node_load_system(full_system_path)
    @warn "FULL SYSTEM:"
    display(sys_full)

    label_area!(sys_full, [SURROGATE_BUS], "surrogate")

    ##########GENERATE TRAIN DATA FROM FAULT##############################
    #perturbations = all_line_trips(sys_full, 0.5)
    g = PSY.get_component(PSY.DynamicInjection, sys_full, fault_generator)
    @warn PSY.get_P_ref(g)

    perturbations = [PowerSimulationsDynamics.ControlReferenceChange(0.5, g, :P_ref, 0.55)]

    pvs_data = generate_pvs_data(
        sys_full,
        perturbations,
        GenerateDataParams(tspan = (0.0, 3.0), steps = 300),
        "surrogate",
    )
    g = PSY.get_component(PSY.DynamicInjection, sys_full, fault_generator)
    @warn PSY.get_P_ref(g)


    ##########GENERATE TRAIN DATA FROM PVS COEFFICIENTS###################
    #keep bias and powerflow quantites same, change perturbation. 
    if train_from_coefficients
        pvs_data[1][1].internal_voltage_frequencies = [2 * pi / 3]
        pvs_data[1][1].internal_voltage_coefficients = [(0.001, 0.0)]
        pvs_data[1][1].internal_angle_frequencies = [2 * pi / 3]
        pvs_data[1][1].internal_angle_coefficients = [(0.0, 0.0)]
    end

    sys_train = create_surrogate_training_system(sys_full, "surrogate", pvs_data)
    @warn "TRAIN SYSTEM:"
    display(sys_train)

    d = generate_train_data(sys_train, GenerateDataParams(tspan = (0.0, 6.0), steps = 300))

    PSY.to_json(
        sys_train,
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json"),
        force = true,
    )

    Serialization.serialize(
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "data"),
        d,
    )

    p = TrainParams(
        base_path = path,
        #curriculum = "progressive"
        curriculum_timespans = [
            (tspan = (0.0, 0.05), batching_sample_factor = 1.0),    #TODO - implement 
            #(tspan = (0.0, 1.0), batching_sample_factor = 0.9),
        ],
        optimizer = (
            sensealg = "Zygote",
            primary = "Adam",
            primary_η = 0.001,
            adjust = "nothing",
            adjust_η = 0.0,
        ),
        model_node = (
            type = "dense",
            n_layer = 1,
            width_layers = 10,
            activation = "hardtanh",
            normalization = "default",
            initialization = "default",
            exogenous_input = "V",
        ),
        dynamic_solver = (
            solver = "Tsit5",
            tols = (1e-6, 1e-6),
            sensealg = "InterpolatingAdjoint",
            maxiters = 1e6,
        ),
        steady_state_solver = (
            solver = "Tsit5",
            tols = (1e-4, 1e-4),        #High tolerance -> standard NODE with initializer and observation 
            sensealg = "InterpolatingAdjoint",
        ),
        loss_function = (
            component_weights = (A = 1.0, B = 1.0, C = 1.0),
            type_weights = (rmse = 1.0, mae = 0.0),
            scale = "default",
        ),
    )
    display(p)

    status = train(p)

    @test status
    input_param_file = joinpath(path, "input_data", "input_test2.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    visualize_training(input_param_file, visualize_level = 4)
    animate_training(input_param_file, skip_frames = 1)
    p.sensealg = "Zygote"
    p.train_id = "train_instance_2"
    status = train(p)
    @test status
    input_param_file = joinpath(path, "input_data", "input_test2.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    visualize_training(input_param_file, visualize_level = 1)
    animate_training(input_param_file, skip_frames = 1)

    a = generate_summary(joinpath(path, "output_data"))
    print_high_level_output_overview(a, path)
finally
    @info("removing test files")
    #rm(path, force = true, recursive = true)
end
