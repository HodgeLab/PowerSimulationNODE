include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
include(joinpath(TEST_FILES_DIR, "scripts", "build_full_system.jl"))

path = (joinpath(pwd(), "test-train-dir"))
!isdir(path) && mkdir(path)

try
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    mkpath(joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME))
    mkpath(joinpath(path, "scripts"))

    full_system_path =
        joinpath(path, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "full_system.json")
    yaml_path = joinpath(path, "scripts", "config.yml")
    SURROGATE_BUS = 16

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

    @warn "Rebuilding input data files"
    sys_full = node_load_system(full_system_path)

    pvs_data = fault_data_generator(yaml_path, full_system_path)
    sys_pvs = build_pvs(pvs_data)
    label_area!(sys_full, [SURROGATE_BUS], "surrogate")
    @assert check_single_connecting_line_condition(sys_full)
    sys_surr = remove_area(sys_full, "1")
    sys_train = build_train_system(sys_surr, sys_pvs, "surrogate")
    PSY.to_json(
        sys_train,
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json"),
        force = true,
    )
    @warn joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json")
    d = generate_train_data(
        sys_train,
        NODETrainDataParams(ode_model = "vsm"),
        SURROGATE_BUS,
        inv_case78("aa"),
    )

    Serialization.serialize(
        joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "data"),
        d,
    )

    #TODO - need additional tests for ode_model = "vsm" 
    p = NODETrainParams(
        base_path = path,
        ode_model = "none",
        solver = "Tsit5",
        solver_tols = (1e-6, 1e-3),
        solver_sensealg = "InterpolatingAdjoint_checkpointing",
        input_PQ = false,
        node_unobserved_states = 8,
        node_activation = "relu",
        sensealg = "ForwardDiff",
        learn_initial_condition_unobserved_states = true,
        node_layers = 2,
        node_width = 4,
        groupsize_faults = 1,
        verify_psid_node_off = false,
        maxiters = 10,
        optimizer_η = 0.001,
        output_mode = 3,
        output_mode_skip = 2,
        node_input_scale = 1.0,
        training_groups = [
            (
                tspan = (0.0, 1.0),
                shoot_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                multiple_shoot_continuity_term = (100.0, 100.0),
                batching_sample_factor = 1.0,
            ),
        ],
        node_state_inputs = [],
        #= node_state_inputs = [
            ("gen1", :ir_filter),
            ("gen1", :ii_filter),
            ("gen1", :θ_pll),
            ("gen1", :ϕq_ic),
            ("gen1", :ϕd_ic),
        ], =#
    )
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
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
