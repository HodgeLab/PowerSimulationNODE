
function pvs_simple(source)
    return PeriodicVariableSource(
        name = get_name(source),
        R_th = get_R_th(source),
        X_th = get_X_th(source),
        internal_voltage_bias = 1.0,
        internal_voltage_frequencies = [2 * pi],
        internal_voltage_coefficients = [(0.05, 0.0)],
        internal_angle_bias = 0.0,
        internal_angle_frequencies = [2 * pi],
        internal_angle_coefficients = [(0.05, 0.0)],
    )
end

@testset "Compare PSID and Training Surrogate - PVS" begin
    #READ SYSTEM WITHOUT GENS 
    sys = System(joinpath(TEST_FILES_DIR, "system_data/2bus_nogens.raw"))
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))

    #ADD TWO SOURCES 
    for b in get_components(Bus, sys)
        if get_number(b) == 1
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )
            add_component!(sys, source)
        end
        if get_number(b) == 2
            l = PowerLoad(              #Can't have a source because we don't get any ΔV at the surrogate bus m
                name = "Load_2",
                available = true,
                base_power = 100.0,
                model = LoadModels.ConstantImpedance,
                bus = b,
                active_power = 1.0,
                reactive_power = 0.1,
                max_active_power = 2.0,
                max_reactive_power = 2.0,
            )
            add_component!(sys, l)
        end
    end
    solve_powerflow(sys)["flow_results"]
    #SERIALIZE TO SYSTEM
    to_json(sys, joinpath(pwd(), "test", "system_data", "test.json"), force = true)

    #DEFAULT PARAMETERS FOR THAT SYSTEM
    p = TrainParams(
        base_path = joinpath(pwd(), "test"),
        surrogate_buses = [2],
        model_node = (
            type = "dense",
            n_layer = 1,
            width_layers = 4,
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
                solver = "Rodas4",
                formulation = "MassMatrix",
                solver_tols = (1e-6, 1e-6),
                all_lines_dynamic = true,
            ),
            system = "reduced",
        ),
        system_path = joinpath(pwd(), "test", "system_data", "test.json"),
        rng_seed = 2,
        dynamic_solver = (solver = "Rodas4", tols = (1e-6, 1e-6), maxiters = 1e5),
    )

    build_subsystems(p)
    mkpath(joinpath(p.base_path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    generate_train_data(p)
    Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
    train_dataset = Serialization.deserialize(p.train_data_path)
    scaling_extrema = PowerSimulationNODE.calculate_scaling_extrema(train_dataset)
    sys_validation = System(p.surrogate_system_path)
    sys_train = System(p.train_system_path)
    exs = PowerSimulationNODE._build_exogenous_input_functions(p.train_data, train_dataset)
    v0 = [
        train_dataset[1].surrogate_real_voltage[1],
        train_dataset[1].surrogate_imag_voltage[1],
    ]
    i0 = [train_dataset[1].branch_real_current[1], train_dataset[1].branch_imag_current[1]]

    tsteps = train_dataset[1].tsteps
    tstops = train_dataset[1].tstops
    Vr1_flux = [exs[1](t)[1] for t in tsteps] #The output of ex() is Vr,Vi 
    Vi1_flux = [exs[1](t)[2] for t in tsteps]
    Vm1_flux = sqrt.(Vr1_flux .^ 2 .+ Vi1_flux .^ 2)
    θ1_flux = atan.(Vi1_flux ./ Vr1_flux)
    p3 = plot(tsteps, Vm1_flux, label = "Vm1 - flux")
    p4 = plot(tsteps, θ1_flux, label = "θ1 - flux")
    connecting_branches = Serialization.deserialize(p.connecting_branch_names_path)
    #INSTANTIATE BOTH TYPES OF SURROGATES 
    train_surrogate = PowerSimulationNODE.instantiate_surrogate_flux(
        p,
        1,
        scaling_extrema,
        connecting_branches,
        sys_train,
    ) #Add connecting branches 
    psid_surrogate =
        PowerSimulationNODE.instantiate_surrogate_psid(p, 1, scaling_extrema, "test-source")
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

    #SET THE PARAMETERS OF THE PSID SURROGATE FROM THE FLUX ONE 
    θ, _ = Flux.destructure(train_surrogate)
    PSIDS.set_initializer_parameters!(psid_surrogate, θ[1:(train_surrogate.len)])
    PSIDS.set_node_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)],
    )
    PSIDS.set_observer_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + train_surrogate.len2 + 1):end],
    )

    #ADD THE SURROGATE COMPONENT 
    for s in get_components(Source, sys_validation)
        if get_number(get_bus(s)) == 1
            pvs = PeriodicVariableSource(
                name = get_name(s),
                R_th = get_R_th(s),
                X_th = get_X_th(s),
                internal_voltage_bias = 1.0,
                internal_voltage_frequencies = [2 * pi],
                internal_voltage_coefficients = [(0.05, 0.0)],
                internal_angle_bias = 0.0,
                internal_angle_frequencies = [2 * pi],
                internal_angle_coefficients = [(0.05, 0.0)],
            )
            add_component!(sys_validation, pvs, s)
        end
        if get_number(get_bus(s)) == 2
            set_name!(psid_surrogate, get_name(s))
            add_component!(sys_validation, psid_surrogate, s)
        end
    end

    #SIMULATE AND PLOT
    sim = Simulation!(MassMatrixModel, sys_validation, pwd(), (0.0, 1.0))
    show_states_initial_value(sim)
    execute!(sim, Rodas4(), abstol = 1e-9, reltol = 1e-9)
    results = read_results(sim)
    Vm1 = get_voltage_magnitude_series(results, 1)
    θ1 = get_voltage_angle_series(results, 1)
    Ir = get_real_current_branch_flow(results, "BUS 1-BUS 2-i_1")
    Ii = get_imaginary_current_branch_flow(results, "BUS 1-BUS 2-i_1")
    plot!(p3, Vm1, label = "Vm1 - psid")
    plot!(p4, θ1, label = "θ1 - psid")
    plot!(p1, Ir[1], Ir[2] .* -1, label = "real current -psid", legend = :topright)
    plot!(p2, Ii[1], Ii[2] .* -1, label = "imag current -psid", legend = :topright)
    display(plot(p1, p2, p3, p4, size = (1000, 1000)))

    #@test LinearAlgebra.norm(Ir[2] .* -1 .- surrogate_sol.i_series[1, :], Inf) <= 5e-4

    #See the distribution of the parameters
    #= p_params = scatter(θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)], label = "node params")
    scatter!(p_params, θ[1:(train_surrogate.len)], label = "init params")
    scatter!(p_params, θ[(train_surrogate.len + train_surrogate.len2 + 1):end], label = "observe params")
    display(p_params) =#
end

@testset "Compare PSID and Training Surrogate - VStep" begin
    sys = System(joinpath(TEST_FILES_DIR, "system_data/2bus_gen.raw"))
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
    TSPAN = (0.0, 1.0)  #Scaling factors calculated based on training data of this length
    TFAULT = 0.05
    for b in get_components(Bus, sys)
        if get_number(b) == 1
            display(b)
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 1e-6, #TODO- change to PSIDS.SOURCE_R_TH
                X_th = 1e-6, #TODO- change to PSIDS.SOURCE_X_TH
            )
            add_component!(sys, source)
        end
        if get_number(b) == 2
            g = collect(get_components(ThermalStandard, sys))[1]
            dyn_g = inv_darco_droop(get_name(g))
            add_component!(sys, dyn_g, g)
        end
    end

    #SERIALIZE TO SYSTEM
    to_json(sys, joinpath(pwd(), "test", "system_data", "test.json"), force = true)

    #DEFAULT PARAMETERS FOR THAT SYSTEM
    p = TrainParams(
        base_path = joinpath(pwd(), "test"),
        surrogate_buses = [2],
        model_node = (
            type = "dense",
            n_layer = 1,
            width_layers = 4,
            activation = "hardtanh",
            σ2_initialization = 0.0,
        ),
        train_data = (
            id = "1",
            operating_points = PSIDS.SurrogateOperatingPoint[PSIDS.GenerationLoadScale()],
            perturbations = [[
                PSIDS.VStep(source_name = "source_1", t_step = TFAULT, V_step = 1.2),
            ]],
            params = PSIDS.GenerateDataParams(
                solver = "Rodas4",
                tspan = TSPAN,
                formulation = "MassMatrix",
                solver_tols = (1e-6, 1e-6),
                all_lines_dynamic = true,
            ),
            system = "reduced",
        ),
        dynamic_solver = (solver = "Rodas4", tols = (1e-6, 1e-6), maxiters = 1e5),
        system_path = joinpath(pwd(), "test", "system_data", "test.json"),
        rng_seed = 3,
    )

    build_subsystems(p)
    mkpath(joinpath(p.base_path, PowerSimulationNODE.INPUT_FOLDER_NAME))
    sys_validation = System(p.surrogate_system_path)
    sys_train = System(p.train_system_path)

    generate_train_data(p)
    Random.seed!(p.rng_seed) #Seed call usually happens at start of train()
    train_dataset = Serialization.deserialize(p.train_data_path)

    scaling_extrema = PowerSimulationNODE.calculate_scaling_extrema(train_dataset)
    sys_validation = System(p.surrogate_system_path)
    exs = PowerSimulationNODE._build_exogenous_input_functions(p.train_data, train_dataset)
    v0 = [
        train_dataset[1].surrogate_real_voltage[1],
        train_dataset[1].surrogate_imag_voltage[1],
    ]
    i0 = [train_dataset[1].branch_real_current[1], train_dataset[1].branch_imag_current[1]]

    tsteps = train_dataset[1].tsteps
    tstops = train_dataset[1].tstops
    Vr1_flux = [exs[1](t)[1] for t in tsteps]
    Vi1_flux = [exs[1](t)[2] for t in tsteps]
    Vm1_flux = sqrt.(Vr1_flux .^ 2 .+ Vi1_flux .^ 2)
    θ1_flux = atan.(Vi1_flux ./ Vr1_flux)
    p1 = plot(tsteps, Vm1_flux, label = "Vm1 - flux")
    p2 = plot(tsteps, θ1_flux, label = "θ1 - flux")

    #INSTANTIATE BOTH TYPES OF SURROGATES 
    connecting_branches = Serialization.deserialize(p.connecting_branch_names_path)
    #INSTANTIATE BOTH TYPES OF SURROGATES 
    train_surrogate = PowerSimulationNODE.instantiate_surrogate_flux(
        p,
        1,
        scaling_extrema,
        connecting_branches,
        sys_train,
    ) #Add connecting branches 

    #train_surrogate = PowerSimulationNODE.instantiate_surrogate_flux(p, 1, scaling_extrema)
    psid_surrogate =
        PowerSimulationNODE.instantiate_surrogate_psid(p, 1, scaling_extrema, "test-source")
    surrogate_sol = train_surrogate(exs[1], v0, i0, tsteps, tstops)
    p3 = plot(
        surrogate_sol.t_series,
        surrogate_sol.i_series[1, :],
        label = "real current - flux",
    )
    p4 = plot(
        surrogate_sol.t_series,
        surrogate_sol.i_series[2, :],
        label = "imag current - flux",
    )
    p5 = plot(
        surrogate_sol.t_series,   #sqrt.(Vr1_flux .^ 2 .+ Vi1_flux .^ 2)
        sqrt.(surrogate_sol.v_series[1, :] .^ 2 .+ surrogate_sol.v_series[2, :] .^ 2),
        label = "surrogate voltage mag - flux",
    )
    p6 = plot(
        surrogate_sol.t_series,
        atan.(surrogate_sol.v_series[2, :] ./ surrogate_sol.v_series[1, :]),
        label = "surrogate voltage angle - flux",
    )

    #SET THE PARAMETERS OF THE PSID SURROGATE FROM THE FLUX ONE 
    θ, _ = Flux.destructure(train_surrogate)
    PSIDS.set_initializer_parameters!(psid_surrogate, θ[1:(train_surrogate.len)])
    PSIDS.set_node_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)],
    )
    PSIDS.set_observer_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + train_surrogate.len2 + 1):end],
    )

    #ADD THE SURROGATE COMPONENT 
    for s in get_components(Source, sys_validation)
        if get_number(get_bus(s)) == 1
        end
        if get_number(get_bus(s)) == 2
            set_name!(psid_surrogate, get_name(s))
            add_component!(sys_validation, psid_surrogate, s)
        end
    end
    source = collect(get_components(Source, sys_validation))[2]
    p = PSID.SourceBusVoltageChange(TFAULT, source, :V_ref, 1.2)
    #SIMULATE AND PLOT
    sim = Simulation!(
        MassMatrixModel,
        sys_validation,
        pwd(),
        TSPAN,
        p,
        all_lines_dynamic = true,
    )
    show_states_initial_value(sim)
    execute!(sim, Rodas4(), abstol = 1e-6, reltol = 1e-6)
    results = read_results(sim)
    Vm1 = get_voltage_magnitude_series(results, 1)
    Vm2 = get_voltage_magnitude_series(results, 2)
    θ1 = get_voltage_angle_series(results, 1)
    θ2 = get_voltage_angle_series(results, 2)
    Ir = get_state_series(results, ("BUS 1-BUS 2-i_1", :Il_R))  #For algebraic lines -> Ir = get_real_current_branch_flow(results, "BUS 1-BUS 2-i_1")
    Ii = get_state_series(results, ("BUS 1-BUS 2-i_1", :Il_I))#For algebraic lines -> Ii = get_imaginary_current_branch_flow(results, "BUS 1-BUS 2-i_1")

    plot!(p1, Vm1, label = "Vm1 - psid")
    plot!(p2, θ1, label = "θ1 - psid")
    plot!(p3, Ir[1], Ir[2] .* -1, label = "real current -psid", legend = :topright)
    plot!(p4, Ii[1], Ii[2] .* -1, label = "imag current -psid", legend = :topright)
    plot!(p5, Vm2, label = "surrogate voltage mag - psid")
    plot!(p6, θ2, label = "surrogate voltage angle - psid")
    display(plot(p1, p2, p3, p4, p5, p6, size = (1000, 1000)))
    @test true
    #@test LinearAlgebra.norm(Ir[2] .* -1 .- surrogate_sol.i_series[1, :], Inf) <= 5e-4  #TODO - get these at same points

    #See the distribution of the parameters
    #= p_params = scatter(θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)], label = "node params")
    scatter!(p_params, θ[1:(train_surrogate.len)], label = "init params")
    scatter!(p_params, θ[(train_surrogate.len + train_surrogate.len2 + 1):end], label = "observe params")
    display(p_params) =#
end

@testset "Compare gfl response for different angle reference" begin
    sys = System(joinpath(TEST_FILES_DIR, "system_data/3bus_nogens.raw"))
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))
    for b in get_components(Bus, sys)
        if get_number(b) == 1
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )
            add_component!(sys, source)
        end
        if get_number(b) == 2
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )
            add_component!(sys, source)
            pvs = pvs_simple(source)
            add_component!(sys, pvs, source)
        end
        if get_number(b) == 3
            gen = collect(get_components(ThermalStandard, sys))[1]
            #source version             
            #= remove_component!(sys, gen)
            source = Source( name = "source_$(get_name(b))", active_power= 1.0, available = true, reactive_power = 0.0, bus = b, R_th = 0.0, X_th = 5e-6)
            add_component!(sys, source) =#

            #gen version 
            gfl = inv_gfoll(gen)
            add_component!(sys, gfl, gen)
        end
    end
    solve_powerflow!(sys)
    sim = Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0))
    execute!(sim, Rodas5(), saveat = 0.0:0.01:1.0)
    results = read_results(sim)
    Vm3_ref1 = get_voltage_magnitude_series(results, 3)[2]
    θ3_ref1 = get_voltage_angle_series(results, 3)[2]
    Ir23_ref1 = get_real_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Ii23_ref1 = get_imaginary_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Im23_ref1 = sqrt.(Ir23_ref1 .^ 2 .+ Ii23_ref1 .^ 2)

    for b in get_components(Bus, sys)
        if get_number(b) == 1
            set_bustype!(b, PowerSystems.BusTypes.PQ)
        end
        if get_number(b) == 2
            set_bustype!(b, PowerSystems.BusTypes.REF)
            set_angle!(b, 0.0)
        end
    end

    solve_powerflow!(sys)
    sim = Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0))
    execute!(sim, Rodas5(), saveat = 0.0:0.01:1.0)
    results = read_results(sim)
    Vm3_ref2 = get_voltage_magnitude_series(results, 3)[2]
    θ3_ref2 = get_voltage_angle_series(results, 3)[2]
    Ir23_ref2 = get_real_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Ii23_ref2 = get_imaginary_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Im23_ref2 = sqrt.(Ir23_ref2 .^ 2 .+ Ii23_ref2 .^ 2)

    p1 = plot(θ3_ref1, label = "θdevice - ref bus 1")
    plot!(p1, θ3_ref2, label = "θdevice - ref bus 2")
    p2 = plot(Im23_ref1, label = "Im_device - ref bus 1")
    plot!(p2, Im23_ref2, label = "Im_device - ref bus 2")
    #display(plot(p1, p2))

    @test LinearAlgebra.norm(Vm3_ref1 .- Vm3_ref2, Inf) <= 1e-3
    @test LinearAlgebra.norm(θ3_ref1 .- θ3_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Ir23_ref1 .- Ir23_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Ii23_ref1 .- Ii23_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Im23_ref1 .- Im23_ref2, Inf) <= 1e-3
end

#NOTE - once we develop this test we can delete the test above... no need to test the gfl in this package. 
#= @testset "Compare surrogate response for different angle reference" begin
    sys = System(joinpath(TEST_FILES_DIR, "system_data/3bus_nogens.raw"))
    include(joinpath(TEST_FILES_DIR, "system_data/dynamic_components_data.jl"))

    p = TrainParams(
        model_node = (
            type = "dense",
            n_layer = 1,
            width_layers = 4,
            activation = "hardtanh",
            σ2_initialization = 0.05,
        ),
    )
    scaling_extrema = Dict{}(
        "target_max" => [1.0, 1.0],
        "target_min" => [-1.0, -1.0],
        "input_max" => [1.0, 1.0],
        "input_min" => [-1.0, -1.0],
    )
    train_surrogate = PowerSimulationNODE.instantiate_surrogate_flux(p, 1, scaling_extrema)
    psid_surrogate =
        PowerSimulationNODE.instantiate_surrogate_psid(p, 1, scaling_extrema, "test-source")
    θ, _ = Flux.destructure(train_surrogate)
    PSIDS.set_initializer_parameters!(psid_surrogate, θ[1:(train_surrogate.len)])
    PSIDS.set_node_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + 1):(train_surrogate.len + train_surrogate.len2)],
    )
    PSIDS.set_observer_parameters!(
        psid_surrogate,
        θ[(train_surrogate.len + train_surrogate.len2 + 1):end],
    )
    for b in get_components(Bus, sys)
        if get_number(b) == 1
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )
            add_component!(sys, source)
        end
        if get_number(b) == 2
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )
            add_component!(sys, source)
            pvs = pvs_simple(source)
            add_component!(sys, pvs, source)
        end
        if get_number(b) == 3
            gen = collect(get_components(ThermalStandard, sys))[1]
            #source version             
            remove_component!(sys, gen)
            source = Source(
                name = "source_$(get_name(b))",
                active_power = 1.0,
                available = true,
                reactive_power = 0.0,
                bus = b,
                R_th = 0.0,
                X_th = 5e-6,
            )

            add_component!(sys, source)
            set_name!(psid_surrogate, get_name(source))
            add_component!(sys, psid_surrogate, source)

            #gen version 
            #gfl = inv_gfoll(gen)
            #add_component!(sys, gfl, gen)
        end
    end
    solve_powerflow!(sys)
    sim = Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0))
    show_states_initial_value(sim)
    execute!(sim, Rodas5(), saveat = 0.0:0.01:1.0)
    results = read_results(sim)
    Vm3_ref1 = get_voltage_magnitude_series(results, 3)[2]
    θ3_ref1 = get_voltage_angle_series(results, 3)[2]
    Ir23_ref1 = get_real_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Ii23_ref1 = get_imaginary_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Im23_ref1 = sqrt.(Ir23_ref1 .^ 2 .+ Ii23_ref1 .^ 2)

    for b in get_components(Bus, sys)
        if get_number(b) == 1
            set_bustype!(b, PowerSystems.BusTypes.PQ)
        end
        if get_number(b) == 2
            set_bustype!(b, PowerSystems.BusTypes.REF)
            set_angle!(b, 0.0)
        end
    end

    solve_powerflow!(sys)
    sim = Simulation!(MassMatrixModel, sys, pwd(), (0.0, 1.0))
    execute!(sim, Rodas5(), saveat = 0.0:0.01:1.0)
    results = read_results(sim)
    Vm3_ref2 = get_voltage_magnitude_series(results, 3)[2]
    θ3_ref2 = get_voltage_angle_series(results, 3)[2]
    Ir23_ref2 = get_real_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Ii23_ref2 = get_imaginary_current_branch_flow(results, "BUS 2-BUS 3-i_1")[2]
    Im23_ref2 = sqrt.(Ir23_ref2 .^ 2 .+ Ii23_ref2 .^ 2)

    p1 = plot(θ3_ref1, label = "θdevice - ref bus 1")
    plot!(p1, θ3_ref2, label = "θdevice - ref bus 2")
    p2 = plot(Im23_ref1, label = "Im_device - ref bus 1", legend = :bottomright)
    plot!(p2, Im23_ref2, label = "Im_device - ref bus 2", legend = :bottomright)
    #display(plot(p1, p2))

    @test LinearAlgebra.norm(Vm3_ref1 .- Vm3_ref2, Inf) <= 1e-3
    @test LinearAlgebra.norm(θ3_ref1 .- θ3_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Ir23_ref1 .- Ir23_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Ii23_ref1 .- Ii23_ref2, Inf) >= 1e-2
    @test LinearAlgebra.norm(Im23_ref1 .- Im23_ref2, Inf) <= 1e-3
end
 =#
