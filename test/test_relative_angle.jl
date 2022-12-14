#The goal of this test is to make sure the surrogate model does not depend on the absolute value of voltage angle.
#A test is shown with a gfl inverter and then repeated for each surrogate model.
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

@testset "Compare gfl response for different angle reference" begin
    sys = System(joinpath(TEST_FILES_DIR, "system_data/3bus_nogens.raw"))   #3 buses connected by 2 branches
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
            gfl = inv_gfoll(gen)
            add_component!(sys, gfl, gen)
        end
    end
    node_run_powerflow!(sys)
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

    node_run_powerflow!(sys)
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

@testset "Compare SteadyStateNodeObs response for different angle reference" begin
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
        model_observation = (
            type = "dense",
            n_layer = 0,
            width_layers = 4,
            activation = "hardtanh",
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
        end
    end
    node_run_powerflow!(sys)
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

    node_run_powerflow!(sys)
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

@testset "Compare SteadyStateNode response for different angle reference" begin
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
        model_observation = (
            type = "DirectObservation",
            n_layer = 0,
            width_layers = 4,
            activation = "hardtanh",
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
    #=     PSIDS.set_observer_parameters!(
            psid_surrogate,
            θ[(train_surrogate.len + train_surrogate.len2 + 1):end],
        ) =#
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
        end
    end
    node_run_powerflow!(sys)
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

    node_run_powerflow!(sys)
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
