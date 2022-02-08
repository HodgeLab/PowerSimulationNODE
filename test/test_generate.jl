include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
include(joinpath(TEST_FILES_DIR, "scripts", "build_full_system.jl"))

path = (joinpath(pwd(), "test-train-dir"))
!isdir(path) && mkdir(path)

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
@warn joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_FOLDER_NAME, "system.json")    #SHOULD FAIL AT INV_CASE78
d = generate_train_data(
    sys_train,
    NODETrainDataParams(ode_model = "none"),
    SURROGATE_BUS,
    inv_case78("aa"),
)

Serialization.serialize(joinpath(path, PowerSimulationNODE.INPUT_FOLDER_NAME, "data"), d)
