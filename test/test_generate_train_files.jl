
mkpath(joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_FOLDER_NAME))

include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))

train_data_path = joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_FOLDER_NAME, "data.json")
train_system_path = joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json")
full_system_path = joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "full_system.json")
yaml_path = joinpath(TEST_FILES_DIR, "scripts", "config.yml")
SURROGATE_BUS = 16

@warn "Rebuilding full system"
include(joinpath(TEST_FILES_DIR, "scripts/build_full_system.jl"))


@warn "Rebuilding input data files"
@warn full_system_path
sys_full = node_load_system(full_system_path)
pvs_data = fault_data_generator(yaml_path)
sys_pvs = build_pvs(pvs_data)
label_area!(sys_full, [SURROGATE_BUS], "surrogate")
@assert check_single_connecting_line_condition(sys_full)
sys_surr = remove_area(sys_full, "1")
sys_train = build_train_system(sys_surr, sys_pvs, "surrogate")
to_json(sys_train, joinpath(TEST_FILES_DIR,PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json"), force = true)
@warn  joinpath(TEST_FILES_DIR,PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json")
d = generate_train_data(sys_train, NODETrainDataParams(ode_model = "vsm"), SURROGATE_BUS, inv_case78("aa"))
@warn d 
@warn typeof(d)
serialize(d, joinpath(TEST_FILES_DIR,PowerSimulationNODE.INPUT_FOLDER_NAME, "data.json"))

#TODO: Add real @test statements 
@test 1 == 1 