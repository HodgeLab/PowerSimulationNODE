
@testset "Compare GFM in Flux and PSID - FrequencyChirp" begin
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
    to_json(sys, joinpath(pwd(), "test", "system_data", "test.json"), force = true)

    #DEFAULT PARAMETERS FOR THAT SYSTEM
    p = TrainParams(
        base_path = joinpath(pwd(), "test"),
        surrogate_buses = [2],
        model_params = PSIDS.GFMParams(name = "source_surrogate"),
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
                    V_amp = 0.02,
                    ω_amp = 0.02,
                ),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas5",
                solver_tols = (reltol = 1e-6, abstol = 1e-6),
                tspan = (0.0, 1.0),
                tstops = 0.0:0.001:1.0,
                tsave = 0.0:0.001:1.0,
                formulation = "MassMatrix",
                all_branches_dynamic = false,
                all_lines_dynamic = false,
                seed = 2,
            ),
            system = "reduced", #Use the reduced system to generate the data!
        ),
        system_path = joinpath(pwd(), "test", "system_data", "test.json"),
        rng_seed = 4,
        dynamic_solver = (
            solver = "Rodas5",
            reltol = 1e-6,
            abstol = 1e-6,
            maxiters = 1e5,
            force_tstops = true,
        ),
    )

    build_subsystems(p)
    mkpath(joinpath(p.base_path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    generate_train_data(p)

    Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
    train_dataset = Serialization.deserialize(p.train_data_path)
    sys_train = System(p.train_system_path)
    exs = PowerSimulationNODE._build_exogenous_input_functions(p.train_data, train_dataset)
    v0 = [
        train_dataset[1].surrogate_real_voltage[1],
        train_dataset[1].surrogate_imag_voltage[1],
    ]
    i0 = [train_dataset[1].real_current[1], train_dataset[1].imag_current[1]]

    tsteps = train_dataset[1].tsteps
    tstops = train_dataset[1].tstops
    Vr1_flux = [exs[1](t)[1] for t in tsteps]
    Vi1_flux = [exs[1](t)[2] for t in tsteps]
    Vm1_flux = sqrt.(Vr1_flux .^ 2 .+ Vi1_flux .^ 2)
    θ1_flux = atan.(Vi1_flux, Vr1_flux)
    p3 = plot(tsteps, Vm1_flux, label = "Vm1 - flux")
    p4 = plot(tsteps, θ1_flux, label = "θ1 - flux")

    data_collection_location = Serialization.deserialize(p.data_collection_location_path)[2]

    #INSTANTIATE BOTH TYPES OF SURROGATES 
    train_surrogate =
        PowerSimulationNODE.instantiate_surrogate_flux(p, p.model_params, train_dataset)

    surrogate_sol = train_surrogate(exs[1], v0, i0, tsteps, tstops)

    p1 = plot(
        surrogate_sol.t_series,
        surrogate_sol.i_series[1, :],
        label = "real current - flux",
    )
    p2 = plot(
        surrogate_sol.t_series,
        surrogate_sol.i_series[2, :],
        label = "imag current - flux",
    )

    b = collect(get_components(Bus, sys_train))[1]

    #Add a source to attach the surrogate to
    source_surrogate = Source(
        name = "source_surrogate",
        active_power = -1.0,
        available = true,
        reactive_power = -0.1,
        bus = b,
        R_th = 5e-6,
        X_th = 5e-6,
    )
    add_component!(sys_train, source_surrogate)

    θ, _ = Flux.destructure(train_surrogate)

    PowerSimulationNODE.add_surrogate_psid!(sys_train, p.model_params, train_dataset)
    PowerSimulationNODE.parameterize_surrogate_psid!(sys_train, θ, p.model_params)
    display(sys_train)

    #Add the  Frequency Chirp
    for s in get_components(Source, sys_train)
        if get_name(s) == "source_1"
            chirp = FrequencyChirpVariableSource(
                name = get_name(s),
                R_th = get_R_th(s),
                X_th = get_X_th(s),
                ω1 = 2 * pi * 3,
                ω2 = 2 * pi * 3,
                tstart = 0.1,
                N = 0.5,
                V_amp = 0.02,
                ω_amp = 0.02,
            )
            add_component!(sys_train, chirp, s)
        end
    end

    #Remove the true model (the StandardLoad)
    for P in get_components(StandardLoad, sys_train)
        remove_component!(sys_train, P)
    end

    #Match the operating point by defining a dummy dataset with initial current and voltage
    b = get_component(Bus, sys_train, "BUS 2")
    Vm0 = PSY.get_magnitude(b)
    θ0 = PSY.get_angle(b)
    Vr0 = Vm0 * cos(θ0)
    Vi0 = Vm0 * sin(θ0)
    Ir0, Ii0 = PowerSimulationNODE.PQV_to_I(-1.0, -0.1, [Vr0, Vi0])
    data_aux = SteadyStateNODEData(;
        real_current = [Ir0],
        imag_current = [Ii0],
        surrogate_real_voltage = [Vr0],
        surrogate_imag_voltage = [Vi0],
    )
    PSIDS.match_operating_point(sys_train, data_aux, p.model_params)

    #Set reactive power of the Chirp Source to be 0.1
    set_reactive_power!(get_component(Source, sys_train, "source_1"), 0.1)

    #SIMULATE AND PLOT
    sim = Simulation!(
        MassMatrixModel,
        sys_train,
        pwd(),
        (0.0, 1.0),
        frequency_reference = ConstantFrequency(),
    )
    show_states_initial_value(sim)

    execute!(sim, Rodas5(), saveat = 0.0:0.001:1.0, abstol = 1e-9, reltol = 1e-9)
    results = read_results(sim)
    Vm2 = get_voltage_magnitude_series(results, 2)
    θ2 = get_voltage_angle_series(results, 2)
    Ir = get_real_current_series(results, "source_1")
    Ii = get_imaginary_current_series(results, "source_1")
    #Ir = get_state_series(results, ("source_surrogate", :δ))
    #Ii = get_state_series(results, ("source_surrogate", :ω))
    plot!(p3, Vm2, label = "Vm2 - psid")
    plot!(p4, θ2, label = "θ2 - psid")
    #NOTE: i_surrogate = - i_source
    plot!(p1, Ir[1], -1 * Ir[2], label = "real current -psid", legend = :topright)
    plot!(p2, Ii[1], -1 * Ii[2], label = "imag current -psid", legend = :topright)
    display(plot(p1, p2, p3, p4, size = (1000, 1000), title = "compare_GFM"))

    @test LinearAlgebra.norm(Ir[2] .* -1 .- surrogate_sol.i_series[1, :], Inf) <= 0.00026
    @test LinearAlgebra.norm(Ii[2] .* -1 .- surrogate_sol.i_series[2, :], Inf) <= 0.00021
    #See the distribution of the parameters
    #= p_params = scatter(θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)], label = "node params")
    scatter!(p_params, θ[1:(train_surrogate.len)], label = "init params")
    scatter!(p_params, θ[(train_surrogate.len + train_surrogate.len2 + 1):end], label = "observe params")
    display(p_params) =#
end
