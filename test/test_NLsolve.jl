#There are some differences in the initialization in PSID and in Flux... 
#In Flux, we test that the actual current is correct. 
#In PSID we check that the internal current (which has a larger range) is correct. -->TODO: loosen this requirement and check condition after scaling.
#This possibly accounts for the difference in the initial conditions that are found in this test 
#TODO - Need a way to test if Simulation found a valid initial condition but didn't completely fail to build (see note on testing log messages below)
@testset "Compare NLsolve convergence (flux vs psid)" begin
    hidden_states = [5, 10, 15, 20]
    for h in hidden_states
        p = TrainParams(
            base_path = joinpath(pwd(), "test"),
            surrogate_buses = [2],
            model_dynamic = (
                type = "dense",
                hidden_states = h,
                n_layer = 2,
                width_layers = 10,
                activation = "hardtanh",
                σ2_initialization = 0.01,
            ),
            train_data = (
                id = "1",
                operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
                perturbations = [[
                    PSIDS.PVS(
                        source_name = "source_1",
                        internal_voltage_frequencies = [2 * pi],
                        internal_voltage_coefficients = [(0.05, 0.0)],
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
            dynamic_solver = (
                solver = "Rodas5",
                reltol = 1e-6,
                abstol = 1e-6,
                maxiters = 1e5,
            ),
        )

        Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
        #These extrema come from the dataset...
        scaling_extrema = Dict{}(
            "input_min" => [0.8, 0.0],
            "input_max" => [1.0, 0.2],
            "target_min" => [0.45, -0.05],
            "target_max" => [0.55, 0.05],
        )

        train_surrogate =
            PowerSimulationNODE.instantiate_surrogate_flux(p, 1, scaling_extrema)
        θ, _ = Flux.destructure(train_surrogate)
        n_parameters = length(θ)
        θ_ranges = Dict{String, UnitRange{Int64}}(
            "initializer_range" => 1:(train_surrogate.len),
            "node_range" =>
                (train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2),
            "observation_range" =>
                (train_surrogate.len + train_surrogate.len2 + 1):n_parameters,
        )
        psid_surrogate = PowerSimulationNODE.instantiate_surrogate_psid(
            p,
            1,
            scaling_extrema,
            "test-source",
        )

        if typeof(psid_surrogate) == PSIDS.SteadyStateNODEObs
            PSIDS.set_initializer_parameters!(
                psid_surrogate,
                θ[θ_ranges["initializer_range"]],
            )
            PSIDS.set_node_parameters!(psid_surrogate, θ[θ_ranges["node_range"]])
            PSIDS.set_observer_parameters!(psid_surrogate, θ[θ_ranges["observation_range"]])
        elseif typeof(psid_surrogate) == PSIDS.SteadyStateNODE
            PSIDS.set_initializer_parameters!(
                psid_surrogate,
                θ[θ_ranges["initializer_range"]],
            )
            PSIDS.set_node_parameters!(psid_surrogate, θ[θ_ranges["node_range"]])
        end
        display(psid_surrogate)
        #WANT TO TEST IF A STEADY STATE CONDITION WAS FOUND FOR THESE CONDITIONS 
        Vr0 = 0.89
        Vi0 = 0.11
        Ir0 = 0.51
        Ii0 = 0.01
        Vmag = sqrt(Vr0^2 + Vi0^2)
        ang = atan(Vi0 / Vr0)
        V = Vr0 + im * Vi0
        I = Ir0 + im * Ii0
        S = V * conj(I)
        P0 = real(S)
        Q0 = imag(S)
        ex = t -> [Vr0, Vi0]
        surrogate_sol = train_surrogate(ex, [Vr0, Vi0], [Ir0, Ii0], 0:0.1:1.0, 0:0.1:1.0)

        sys = System(100.0)
        b = Bus(
            number = 1,
            name = "01",
            bustype = BusTypes.REF,
            angle = ang,
            magnitude = Vmag,
            voltage_limits = (0.0, 2.0),
            base_voltage = 230,
        )
        add_component!(sys, b)

        s1 = Source(
            name = "source_1",
            available = true,
            bus = b,
            active_power = -P0,
            reactive_power = -Q0,
            R_th = 1e-5,
            X_th = 1e-5,
        )
        add_component!(sys, s1)
        s2 = Source(
            name = "test-source",
            available = true,
            bus = b,
            active_power = P0,
            reactive_power = Q0,
            R_th = 1e-5,
            X_th = 1e-5,
        )
        add_component!(sys, s2)
        add_component!(sys, psid_surrogate, s2)

        #NOTE: was unable to test logs that are generated by Simulation!(...)
        #@test_logs (:warn, "Initialization in SteadyStateNODEObs failed") match_mode=:any Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0)) 
        sim = Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0))

        if sim.status == PSID.BUILT    #Note: just because sim.status == PSID.BUILT does not guarantee an initialization to tolerance took place (see note above about logging)
            psid_converged = true
        else
            psid_converged = false
        end

        @test surrogate_sol.converged
        @test psid_converged

        #Display the initial conditions for flux and psid surrogates (not exactly the same, expected for untrained surrogate?)
        show_states_initial_value(sim)
        @warn "value of states that satisfy NLsolve in flux surrogate $(surrogate_sol.r0)"
    end
end
