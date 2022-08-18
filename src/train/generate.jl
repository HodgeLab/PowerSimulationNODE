"""
    function build_subsystems(p::TrainParams)  

Build `training_system` and `surrogate_system` based on `p.system_path` and `p.surrogate_buses` and serialize them to the respective paths in `p`.
"""
function build_subsystems(p::TrainParams)
    sys_full = node_load_system(p.system_path)
    PSY.solve_powerflow!(sys_full)

    sys_train, connecting_branches =
        PSIDS.create_subsystem_from_buses(sys_full, p.surrogate_buses)
    non_surrogate_buses =
        PSY.get_number.(
            PSY.get_components(
                PSY.Bus,
                sys_full,
                x -> PSY.get_number(x) ∉ p.surrogate_buses,
            ),
        )
    sys_validation, _ = PSIDS.create_subsystem_from_buses(sys_full, non_surrogate_buses)

    #Serialize surrogate system
    PSY.to_json(sys_validation, p.surrogate_system_path, force = true)
    #Serialize train system 
    PSY.to_json(sys_train, p.train_system_path, force = true)
    #Serialize connecting branches
    Serialization.serialize(p.connecting_branch_names_path, connecting_branches)
end

"""
    function generate_train_data(p::TrainParams)  

Generate the train data and serialize to the path in `p`.
"""
function generate_train_data(p::TrainParams)
    sys_full = node_load_system(p.system_path)
    sys_train = node_load_system(p.train_system_path)
    sys_validation = node_load_system(p.surrogate_system_path)
    connecting_branches = Serialization.deserialize(p.connecting_branch_names_path)

    if p.train_data.system == "reduced"
        train_data = PSIDS.generate_surrogate_data(
            sys_train,   #sys_main
            sys_validation,   #sys_aux
            p.train_data.perturbations,
            p.train_data.operating_points,
            PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
            p.train_data.params,
        )
    elseif p.train_data.system == "full"
        train_data = PSIDS.generate_surrogate_data(
            sys_full,   #sys_main
            sys_validation,   #sys_aux
            p.train_data.perturbations,
            p.train_data.operating_points,
            PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
            p.train_data.params,
        )
    else
        @error "invalid parameter for the system to generate train data (should be reduced or full)"
    end

    Serialization.serialize(p.train_data_path, train_data)
end

"""
    function generate_validation_data(p::TrainParams)  

Generate the validation data and serialize to the path in `p`.
"""
function generate_validation_data(p::TrainParams)   #generate the validation data and serialize to path from params
    sys_full = node_load_system(p.system_path)
    sys_validation = node_load_system(p.surrogate_system_path)
    connecting_branches = Serialization.deserialize(p.connecting_branch_names_path)

    validation_data = PSIDS.generate_surrogate_data(
        sys_full,   #sys_main
        sys_validation,  #sys_aux
        p.validation_data.perturbations,
        p.validation_data.operating_points,
        PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        p.validation_data.params,
    )

    Serialization.serialize(p.validation_data_path, validation_data)
end

"""
    function generate_test_data(p::TrainParams)  

Generate the test data and serialize to the path in `p`.
"""
function generate_test_data(p::TrainParams)         #generate the test data and serialize to path from params
    sys_full = node_load_system(p.system_path)
    sys_validation = node_load_system(p.surrogate_system_path)
    connecting_branches = Serialization.deserialize(p.connecting_branch_names_path)

    test_data = PSIDS.generate_surrogate_data(
        sys_full,   #sys_main
        sys_validation,  #sys_aux
        p.test_data.perturbations,
        p.test_data.operating_points,
        PSIDS.SteadyStateNODEDataParams(connecting_branch_names = connecting_branches),
        p.test_data.params,
    )

    Serialization.serialize(p.test_data_path, test_data)
end
