
#Goal: 
#For a set of parameters, generate the training data, instantiate the surrogate, and see if the forward pass of the surrogate finds a steady state solution. 
hidden_states = [10, 20, 40]
for h in hidden_states
    p = TrainParams(
        base_path = joinpath(pwd(), "test"),
        surrogate_buses = [2],
        hidden_states = h,
        model_node = (
            type = "dense",
            n_layer = 2,
            width_layers = 10,
            activation = "hardtanh",
            σ2_initialization = 0.0,
        ),
        train_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations = [[
                PSIDS.PVS(
                    source_name = "source_1",
                    internal_voltage_frequencies = [2 * pi],
                    internal_voltage_coefficients = [(0.05, 0.0)],  #if you make this too large you get distortion in the sine wave from large currents? 
                    internal_angle_frequencies = [2 * pi],
                    internal_angle_coefficients = [(0.05, 0.0)],
                ),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                formulation = "MassMatrix",
                solver_tols = (reltol = 1e-6, abstol = 1e-6),
                all_lines_dynamic = true,
            ),
            system = "reduced",
        ),
        system_path = joinpath(pwd(), "test", "system_data", "test.json"),
        rng_seed = 4,
        dynamic_solver = (solver = "Rodas5", reltol = 1e-6, abstol = 1e-6, maxiters = 1e5),
    )

    @warn p.scaling_limits
    #build_subsystems(p)
    mkpath(joinpath(p.base_path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    #generate_train_data(p)
    Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
    #train_dataset = Serialization.deserialize(p.train_data_path)
    scaling_extrema = Dict{}(
        "input_min" => [-1.0, -1.0],
        "input_max" => [1.0, 1.0],
        "target_min" => [-0.1, -0.1],#"target_min" => [-1.0, -1.0],
        "target_max" => [0.0, 0.0],#"target_max" => [1.0, 1.0],
    )

    #scaling_extrema = PowerSimulationNODE.calculate_scaling_extrema(train_dataset)
    sys_validation = System(p.surrogate_system_path)
    sys_train = System(p.train_system_path)
    # exs = PowerSimulationNODE._build_exogenous_input_functions(p.train_data, train_dataset)
    ex = t -> [1.0, 0.0]

    train_surrogate = PowerSimulationNODE.instantiate_surrogate_flux(p, 1, scaling_extrema) #Add connecting branches 
    #WANT TO TEST IF A STEADY STATE CONDITION WAS FOUND 
    surrogate_sol = train_surrogate(ex, [0.9, 0.1], [-0.05, -0.05], 0:0.1:1.0, 0:0.1:1.0)
    display(scatter(surrogate_sol.res))
    @error h, surrogate_sol.converged
end
@assert false
