raw_file_path =
    joinpath(TEST_FILES_DIR, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME, "case9.m")
base_system_path = joinpath(
    TEST_FILES_DIR,
    PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
    "full_system.json",
)
sys = node_load_system(joinpath(TEST_FILES_DIR, raw_file_path))

#Parses constant power loads from matpower --> replace with constant impedance StandardLoad
for power_load in get_components(PowerLoad, sys)
    standard_load = StandardLoad(
        name = get_name(power_load),
        available = get_available(power_load),
        bus = get_bus(power_load),
        base_power = get_base_power(power_load),
        constant_active_power = 0.0,
        constant_reactive_power = 0.0,
        impedance_active_power = get_active_power(power_load),
        impedance_reactive_power = get_reactive_power(power_load),
        current_active_power = 0.0,
        current_reactive_power = 0.0,
        max_constant_active_power = 0.0,
        max_constant_reactive_power = 0.0,
        max_impedance_active_power = get_max_active_power(power_load),
        max_impedance_reactive_power = get_max_reactive_power(power_load),
        max_current_active_power = 0.0,
        max_current_reactive_power = 0.0,
        services = get_services(power_load),
        dynamic_injector = get_dynamic_injector(power_load),
    )
    remove_component!(sys, power_load)
    add_component!(sys, standard_load)
end

for g in get_components(Generator, sys)
    case_gen = dyn_gen_classic(g)
    add_component!(sys, case_gen, g)
end
node_run_powerflow!(sys)
to_json(sys, base_system_path, force = true)
