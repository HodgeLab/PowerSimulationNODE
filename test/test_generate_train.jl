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

    sys_full = node_load_system(full_system_path)
    @warn "FULL SYSTEM:"
    display(sys_full)

    label_area!(sys_full, [SURROGATE_BUS], "surrogate")

    ##########GENERATE TRAIN DATA FROM FAULT##############################
    g = PSY.get_component(PSY.DynamicInjection, sys_full, fault_generator)
    perturbations = [PowerSimulationsDynamics.ControlReferenceChange(0.5, g, :P_ref, 0.55)]
    pvs_data = generate_pvs_data(
        sys_full,
        perturbations,
        GenerateDataParams(tspan = (0.0, 3.0), steps = 300),
        "surrogate",
    )
    #####################################################################

    ##########GENERATE TRAIN DATA FROM PVS COEFFICIENTS###################
    pvs_coeffs = Dict{Int, Array{NamedTuple}}()
    pvs_coeffs[1] = [(
        internal_voltage_frequencies = [2 * pi / 3],
        internal_voltage_coefficients = [(0.001, 0.0)],
        internal_angle_frequencies = [2 * pi / 3],
        internal_angle_coefficients = [(0.0, 0.0)],
    )]
    pvs_data = generate_pvs_data(sys_full, pvs_coeffs, "surrogate")
    #####################################################################   

    sys_train = create_surrogate_training_system(sys_full, "surrogate", pvs_data)
    @warn "TRAIN SYSTEM:"
    display(sys_train)

    d = generate_train_data(sys_train, GenerateDataParams(tspan = (0.0, 4.0), steps = 400))

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
        maxiters = 5,
        #curriculum = "progressive",
        curriculum_timespans = [
            (tspan = (0.0, 4.0), batching_sample_factor = 1.0),
            # (tspan = (0.0, 2.0), batching_sample_factor = 0.1),    
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
            initialization = "default",
        ),
        input_normalization = (
            x_scale = [1.0, 1.0, 1.0, 1.0],
            x_bias = [0.0, 0.0, 0.0, 0.0],
            exogenous_scale = [1.0, 1.0],
            exogenous_bias = [0.0, 0.0],
        ),
        dynamic_solver = (solver = "Tsit5", tols = (1e-6, 1e-6), maxiters = 1e6),
        steady_state_solver = (
            solver = "Tsit5",
            tols = (1e-4, 1e-4),        #High tolerance -> remove SS layer 
            maxiters = 1e3,
        ),
        loss_function = (
            component_weights = (A = 1.0, B = 1.0, C = 1.0),
            type_weights = (rmse = 1.0, mae = 0.0),
        ),
    )
    display(p)
    status, θ = train(p)
    @test status

    #TODO - test parameter restart
    #=     df_loss = PowerSimulationNODE.read_arrow_file_to_dataframe(joinpath(path, "output_data", p.train_id, "loss"))
        train1_final_loss = df_loss[end,:Loss]
        display(train1_final_loss)
        p.p_start = θ
        status, θ = train(p)
        df_loss = PowerSimulationNODE.read_arrow_file_to_dataframe(joinpath(path, "output_data", p.train_id, "loss"))
        train2_starting_loss = df_loss[1,:Loss]
        @test train1_final_loss == train2_starting_loss  =#

    input_param_file = joinpath(path, "input_data", "input_test2.json")
    PowerSimulationNODE.serialize(p, input_param_file)
    visualize_training(input_param_file, skip = 1)
    animate_training(input_param_file, skip = 1)
    a = generate_summary(joinpath(path, "output_data"))
    p = visualize_summary(a)
    print_high_level_output_overview(a, path)
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
