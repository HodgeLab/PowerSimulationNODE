raw_file_path =
    joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "case9.m")
base_system_path = joinpath(
    TEST_FILES_DIR,
    PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
    "full_system.json",
)
sys = node_load_system(joinpath(TEST_FILES_DIR, raw_file_path))

for g in get_components(Generator, sys)
    case_gen = dyn_gen_classic(g)
    add_component!(sys, case_gen, g)
end
node_run_powerflow!(sys)
to_json(sys, base_system_path, force = true)

#Display useful info about the system 
#= for b in get_components(Bus, sys)
@warn get_number(b), get_bustype(b)
end =#
#= for b in get_components(Branch, sys)
    @warn get_name(b)
end  =#
#= for g in get_components(Generator, sys)
    @warn get_name(g)
    display(g)
end

for g in get_components(Generator, sys)
    @warn get_name(g)
    display(g)
end
for b in get_components(Bus, sys)
    @warn get_number(b), get_bustype(b)
end
 =#
