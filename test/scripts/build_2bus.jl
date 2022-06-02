raw_file_path =
    joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "OMIB.raw")
base_system_path = joinpath(
    TEST_FILES_DIR,
    PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
    "full_system.json",
)
sys = node_load_system(joinpath(TEST_FILES_DIR, raw_file_path))
add_source_to_ref(sys)
gen = [g for g in get_components(Generator, sys)][1]
case_gen = dyn_gen_classic(gen)
add_component!(sys, case_gen, gen)
display(sys)

to_json(sys, base_system_path, force = true)
