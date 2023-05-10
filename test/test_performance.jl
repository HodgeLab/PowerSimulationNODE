@testset "Performance Comparisons" begin
    path = (joinpath(pwd(), "test-compare-dir"))
    !isdir(path) && mkdir(path)
    #READ SYSTEM WITHOUT GENS 
    sys = System(joinpath(TEST_FILES_DIR, "system_data/2bus_nogens.raw"))
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))

    #ADD A SOUCE AND A LOAD
    for b in get_components(Bus, sys)
        if get_number(b) == 1
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.1,
                bus = b,
                R_th = 5e-6,
                X_th = 5e-6,
            )
            add_component!(sys, source)
        end
        if get_number(b) == 2
            l = StandardLoad(
                name = "Load_2",
                available = true,
                base_power = 100.0,
                bus = b,
                impedance_active_power = 1.0,
                impedance_reactive_power = 0.1,
                max_impedance_active_power = 2.0,
                max_impedance_reactive_power = 2.0,
            )
            add_component!(sys, l)
        end
    end

    #SERIALIZE TO SYSTEM
    to_json(sys, joinpath(path, "system_data", "test.json"), force = true)

    #DEFAULT PARAMETERS FOR THAT SYSTEM
    p = TrainParams(
        base_path = path,
        surrogate_buses = [2],
        train_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations = [[
                Chirp(
                    source_name = "source_1",
                    ω1 = 2 * pi * 3,
                    ω2 = 2 * pi * 3,
                    tstart = 0.1,
                    N = 0.5,
                    V_amp = 0.2,
                    ω_amp = 0.2,
                ),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                solver_tols = (reltol = 1e-6, abstol = 1e-6),
                tspan = (0.0, 1.0),
                tstops = 0.0:0.1:1.0,
                tsave = 0.0:0.1:1.0,
                formulation = "MassMatrix",
                all_branches_dynamic = false,
                all_lines_dynamic = false,
                seed = 2,
            ),
            system = "reduced", #Use the reduced system to generate the data!
        ),
        system_path = joinpath(path, "system_data", "test.json"),
        rng_seed = 4,
    )
    build_subsystems(p)
    mkpath(joinpath(p.base_path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    generate_train_data(p)
    Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
    train_dataset = Serialization.deserialize(p.train_data_path)
    exs = PowerSimulationNODE._build_exogenous_input_functions(p.train_data, train_dataset)
    v0 = [
        train_dataset[1].surrogate_real_voltage[1],
        train_dataset[1].surrogate_imag_voltage[1],
    ]
    i0 = [train_dataset[1].real_current[1], train_dataset[1].imag_current[1]]
    tsteps = train_dataset[1].tsteps
    tstops = train_dataset[1].tstops

    steadystate_solvers =
        [(solver = "NewtonRaphson", reltol = 1e-4, abstol = 1e-4, termination = "RelSafe")]
    dynamic_solvers = [
        (
            solver = "Rodas5",
            sensealg = "QuadratureAdjoint",
            reltol = 1e-6,
            abstol = 1e-6,
            maxiters = Int64(1e5),
            force_tstops = true,
        ),
        (
            solver = "TRBDF2",
            sensealg = "QuadratureAdjoint",
            reltol = 1e-6,
            abstol = 1e-6,
            maxiters = Int64(1e5),
            force_tstops = true,
        ),
        (
            solver = "TRBDF2",
            sensealg = "InterpolatingAdjoint",
            reltol = 1e-6,
            abstol = 1e-6,
            maxiters = Int64(1e5),
            force_tstops = true,
        ),
    ]
    models_node = [
        PSIDS.NODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
        PSIDS.NODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 15,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
        PSIDS.NODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 25,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
    ]
    models_ss_node = [
        PSIDS.SteadyStateNODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 5,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
        PSIDS.SteadyStateNODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 15,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
        PSIDS.SteadyStateNODEParams(
            name = "source_surrogate",
            dynamic_layer_type = "dense",
            dynamic_hidden_states = 25,
            dynamic_n_layer = 1,
            dynamic_width_layers_relative_input = -3,
            dynamic_activation = "tanh",
            dynamic_σ2_initialization = 0.1,
        ),
    ]

    times_node = DataFrame(
        dyn_solver = String[],
        dyn_sensealg = String[],
        n_params = Int64[],
        t_forward = Float64[],
        t_grad = Float64[],
    )
    for d in dynamic_solvers
        for m in models_node
            dynamic_solver = PowerSimulationNODE.instantiate_solver(d)
            dynamic_sensealg = PowerSimulationNODE.instantiate_sensealg(d)

            train_surrogate =
                PowerSimulationNODE.instantiate_surrogate_flux(p, m, train_dataset)
            θ, _ = Flux.destructure(train_surrogate)
            forward_1 = @timed train_surrogate(
                exs[1],
                v0,
                i0,
                tsteps,
                tstops,
                dynamic_solver,
                d,
                dynamic_sensealg;
                p_train = θ,
            )
            grad_1 = @timed Zygote.gradient(
                θ -> sum(
                    train_surrogate(
                        exs[1],
                        v0,
                        i0,
                        tsteps,
                        tstops,
                        dynamic_solver,
                        d,
                        dynamic_sensealg;
                        p_train = θ,
                    ).i_series,
                ),
                θ,
            )
            forward_2 = @timed train_surrogate(
                exs[1],
                v0,
                i0,
                tsteps,
                tstops,
                dynamic_solver,
                d,
                dynamic_sensealg;
                p_train = θ,
            )

            grad_2 = @timed Zygote.gradient(
                θ -> sum(
                    train_surrogate(
                        exs[1],
                        v0,
                        i0,
                        tsteps,
                        tstops,
                        dynamic_solver,
                        d,
                        dynamic_sensealg;
                        p_train = θ,
                    ).i_series,
                ),
                θ,
            )

            push!(
                times_node,
                Dict(
                    :dyn_solver => d.solver,
                    :dyn_sensealg => d.sensealg,
                    :n_params => length(θ),
                    :t_forward => forward_2.time,
                    :t_grad => grad_2.time,
                ),
            )
        end
    end

    times_ss_node = DataFrame(
        ss_solver = String[],
        dyn_solver = String[],
        dyn_sensealg = String[],
        n_params = Int64[],
        t_forward = Float64[],
        t_grad = Float64[],
    )
    for s in steadystate_solvers
        for d in dynamic_solvers
            for m in models_ss_node
                steadystate_solver = PowerSimulationNODE.instantiate_steadystate_solver(s)
                dynamic_solver = PowerSimulationNODE.instantiate_solver(d)
                dynamic_sensealg = PowerSimulationNODE.instantiate_sensealg(d)
                train_surrogate =
                    PowerSimulationNODE.instantiate_surrogate_flux(p, m, train_dataset)
                θ, _ = Flux.destructure(train_surrogate)
                forward_1 = @timed train_surrogate(
                    exs[1],
                    v0,
                    i0,
                    tsteps,
                    tstops,
                    steadystate_solver,
                    s,
                    dynamic_solver,
                    d,
                    dynamic_sensealg;
                    p_train = θ,
                )
                grad_1 = @timed Zygote.gradient(
                    θ -> sum(
                        train_surrogate(
                            exs[1],
                            v0,
                            i0,
                            tsteps,
                            tstops,
                            steadystate_solver,
                            s,
                            dynamic_solver,
                            d,
                            dynamic_sensealg;
                            p_train = θ,
                        ).i_series,
                    ),
                    θ,
                )
                forward_2 = @timed train_surrogate(
                    exs[1],
                    v0,
                    i0,
                    tsteps,
                    tstops,
                    steadystate_solver,
                    s,
                    dynamic_solver,
                    d,
                    dynamic_sensealg;
                    p_train = θ,
                )

                grad_2 = @timed Zygote.gradient(
                    θ -> sum(
                        train_surrogate(
                            exs[1],
                            v0,
                            i0,
                            tsteps,
                            tstops,
                            steadystate_solver,
                            s,
                            dynamic_solver,
                            d,
                            dynamic_sensealg;
                            p_train = θ,
                        ).i_series,
                    ),
                    θ,
                )

                push!(
                    times_ss_node,
                    Dict(
                        :ss_solver => s.solver,
                        :dyn_solver => d.solver,
                        :dyn_sensealg => d.sensealg,
                        :n_params => length(θ),
                        :t_forward => forward_2.time,
                        :t_grad => grad_2.time,
                    ),
                )
            end
        end
    end
    @show times_node
    @show times_ss_node
    @test true
end
