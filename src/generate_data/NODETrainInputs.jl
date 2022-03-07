"""
    struct NODETrainInputs

This struct contains the input data needed for training. 
# Fields
- `tsteps::Vector{Float64}`: The time steps corresponding to ground truth data
- `fault_data::Dict{String, Dict{Symbol, Any}}`: The data for each fault including...
"""
struct NODETrainInputs
    tsteps::Vector{Float64}
    n_observable_states::Int
    fault_data::Dict{String, Dict{Symbol, Any}}
end

"""
    mutable struct NODETrainDataParams

# Fields
- `solver::String`: solver used for the generating ground truth data. Valid Values ["Rodas4"]
- `solver_tols:: Tuple{Float64, Float64}`: solver tolerances (abstol, reltol).
- `tspan::Tuple{Float64, Float64}`: timespan of ground truth data.
- `steps::Int64`: Total number of steps for a single fault in ground truth data. The distribution of steps is determined by `tsteps_spacing`.
- `tsteps_spacing:: ["linear"]`: Determines the distribution of save data points. 
- `ode_model:: ["none","vsm"]`: The ode model is used to save initial conditions/parameters for use during training. 
- `base_path::String`: Project directory
- `output_data_path::String`: Relative path from project directory to where outputs are written (inputs to training)
"""
mutable struct NODETrainDataParams
    solver::String
    solver_tols::Tuple{Float64, Float64}
    tspan::Tuple{Float64, Float64}
    steps::Int64
    tsteps_spacing::String
    observable_states::Vector{Tuple{String, Symbol}}
    ode_model::String #"vsm" or "none"
    base_path::String
    output_data_path::String
end

StructTypes.StructType(::Type{NODETrainInputs}) = StructTypes.Struct()

function NODETrainDataParams(;
    solver = "Rodas4",
    solver_tols = (1e-6, 1e-9),
    tspan = (0.0, 1.0),
    steps = 100,
    tsteps_spacing = "linear",
    observable_states = [("gen1", :ir_filter), ("gen1", :ii_filter)],
    ode_model = "none",     #TODO - test generating data with VSM model
    base_path = pwd(),
    output_data_path = joinpath(base_path, "input_data"),
)
    NODETrainDataParams(
        solver,
        solver_tols,
        tspan,
        steps,
        tsteps_spacing,
        observable_states,
        ode_model,
        base_path,
        output_data_path,
    )
end

function NODETrainDataParams(file::AbstractString)
    return JSON3.read(read(file), NODETrainDataParams)
end

#TODO - break up this function for ease of understanding/debugging 
function generate_train_data(sys_train, NODETrainDataParams, SURROGATE_BUS, DynamicInverter)
    tspan = NODETrainDataParams.tspan
    steps = NODETrainDataParams.steps
    if NODETrainDataParams.tsteps_spacing == "linear"
        tsteps = tspan[1]:((tspan[2] - tspan[1]) / steps):tspan[2]
    end
    solver = instantiate_solver(NODETrainDataParams)
    abstol = NODETrainDataParams.solver_tols[1]
    reltol = NODETrainDataParams.solver_tols[2]
    fault_data = Dict{String, Dict{Symbol, Any}}()
    n_observable_states = 0
    for pvs in collect(PSY.get_components(PSY.PeriodicVariableSource, sys_train))
        available_source = activate_next_source!(sys_train)
        set_bus_from_source(available_source)  #Bus voltage is used in power flow. Need to set bus voltage from soure internal voltage of source

        sim_full = PSID.Simulation!(
            PSID.MassMatrixModel,
            sys_train,
            pwd(),
            tspan,
            console_level = PSID_CONSOLE_LEVEL,
            file_level = PSID_FILE_LEVEL,
        )

        PSID.execute!(
            sim_full,
            solver,
            abstol = abstol,
            reltol = reltol,
            reset_simulation = false,
            saveat = tsteps,
            enable_progress_bar = false,
        )
        psid_results_object = PSID.read_results(sim_full)
        active_source =
            collect(PSY.get_components(PSY.Source, sys_train, x -> PSY.get_available(x)))[1]
        ode_data = get_total_current_series(sim_full)
        observables_data = []
        for (i, tuple) in enumerate(NODETrainDataParams.observable_states)
            if i == 1
                observables_data = PSID.get_state_series(psid_results_object, tuple)[2]' #first index contains time 
            else
                observables_data = vcat(
                    observables_data,
                    PSID.get_state_series(psid_results_object, tuple)[2]',
                )
            end
        end
        n_observable_states = size(observables_data, 1)
        @warn "generating data for obserble states:", n_observable_states
        transformer = collect(PSY.get_components(PSY.Transformer2W, sys_train))[1]
        p_network = [
            PSY.get_x(transformer) + PSY.get_X_th(pvs),
            PSY.get_r(transformer) + PSY.get_R_th(pvs),
        ]
        bus_results = PSY.solve_powerflow(sys_train)["bus_results"]
        @info "full system", bus_results
        surrogate_bus_result = bus_results[in([SURROGATE_BUS]).(bus_results.bus_number), :]

        P_pf = surrogate_bus_result[1, :P_gen] / PSY.get_base_power(sys_train)
        Q_pf = surrogate_bus_result[1, :Q_gen] / PSY.get_base_power(sys_train)
        V_pf = surrogate_bus_result[1, :Vm]
        θ_pf = surrogate_bus_result[1, :θ]

        @info collect(PSY.get_components(PSY.Bus, sys_train))
        @warn "P*", P_pf, "Q*", Q_pf, "V*", V_pf, "θ*", θ_pf

        fault_data[PSY.get_name(pvs)] = Dict(
            :p_network => p_network,
            :p_pf => [P_pf, Q_pf, V_pf, θ_pf],
            :ground_truth => ode_data,
            :observable_states => observables_data,
            :psid_results_object => psid_results_object,
            :p_ode => [],
        )
        @warn size(ode_data)
        @warn size(observables_data)

        if NODETrainDataParams.ode_model == "vsm"
            #################### BUILD INITIALIZATION SYSTEM ###############################
            sys_init, p_inv = build_sys_init(sys_train, DynamicInverter) #returns p_inv, the set of average parameters 
            x₀, refs, Vr0, Vi0 = initialize_sys!(sys_init, "gen1")
            Vm, Vθ = Source_to_function_of_time(PSY.get_dynamic_injector(active_source))
            p_ode = vcat(p_inv, refs)
            sim_simp = PSID.Simulation!(
                PSID.MassMatrixModel,
                sys_init,
                pwd(),
                tspan,
                console_level = PSID_CONSOLE_LEVEL,
                file_level = PSID_FILE_LEVEL,
            )
            @info "initialize system power flow",
            PSY.solve_powerflow(sys_init)["flow_results"]
            @info "initialize system power flow",
            PSY.solve_powerflow(sys_init)["bus_results"]
            @debug PSY.show_states_initial_value(sim_simp)
            PSID.execute!(
                sim_simp,
                solver,
                abstol = abstol,
                reltol = reltol,
                initializealg = OrdinaryDiffEq.NoInit(),
                reset_simulation = false,
                saveat = tsteps,
                enable_progress_bar = false,
            )

            avgmodel_data = get_total_current_series(sim_simp)

            fault_data[PSY.get_name(pvs)][:p_ode] = p_ode
            fault_data[PSY.get_name(pvs)][:x₀] = x₀
            fault_data[PSY.get_name(pvs)][:ir_node_off] = avgmodel_data[1, :]
            fault_data[PSY.get_name(pvs)][:ii_node_off] = avgmodel_data[2, :]
        end
    end
    return NODETrainInputs(tsteps, n_observable_states, fault_data)
end

"""
    activate_next_source!(sys::PSY.System)

Either activate the first source if none are available, or make the next source available.
To be used in training surrogate to move on to the next system disturbance. Returns the available source
"""
function activate_next_source!(sys::PSY.System)
    all_sources = collect(PSY.get_components(PSY.Source, sys))
    active_sources = collect(PSY.get_components(PSY.Source, sys, x -> PSY.get_available(x)))
    if length(active_sources) < 1
        @info "no active sources in the system, activating the first source"
        first_source = collect(PSY.get_components(PSY.Source, sys))[1]
        PSY.set_available!(first_source, true)
        return first_source
    elseif length(active_sources) > 1
        @error "more than one active source, cannot determine next active source"
    else
        for (i, source) in enumerate(all_sources)
            if active_sources[1] == source
                PSY.set_available!(all_sources[i], false)
                if source !== last(all_sources)
                    PSY.set_available!(all_sources[i + 1], true)
                    @info "found active source, setting next source active"
                    return all_sources[i + 1]
                else
                    PSY.set_available!(all_sources[1], true)
                    @info "the last source is active, starting over at index 1 "
                    return all_sources[1]
                end
            end
        end
    end
end

"""
Test function description
"""
function build_sys_init(sys_train::PSY.System, DynamicInverter::PSY.DynamicInverter)
    sys_init = deepcopy(sys_train)
    base_power_total = 0.0
    power_total = 0.0
    gfms = collect(PSY.get_components(PSY.ThermalStandard, sys_init))

    for gfm in gfms
        base_power_total += PSY.get_base_power(gfm)
        power_total += PSY.get_base_power(gfm) * PSY.get_active_power(gfm)
        @info base_power_total
        @info power_total
        PSY.remove_component!(sys_init, PSY.get_dynamic_injector(gfm))
        PSY.remove_component!(sys_init, gfm)
    end
    g = PSY.ThermalStandard(
        name = string("gen", string(1)),
        available = true,
        status = true,
        bus = collect(
            PSY.get_components(
                PSY.Bus,
                sys_init,
                x -> PSY.get_bustype(x) == PSY.BusTypes.PV,
            ),
        )[1],
        active_power = power_total / base_power_total, #Only divide base power by n_devices
        reactive_power = 0.0,
        rating = base_power_total,
        active_power_limits = (0.0, 3.0),
        reactive_power_limits = (-3.0, 3.0),
        ramp_limits = nothing,
        operation_cost = PSY.ThreePartCost(nothing),
        base_power = base_power_total,
    )
    PSY.add_component!(sys_init, g)
    inv_typ = DynamicInverter
    PSY.set_name!(inv_typ, PSY.get_name(g))
    PSY.add_component!(sys_init, inv_typ, g)
    p_inv = get_parameters(inv_typ)
    return sys_init, p_inv
end

function initialize_sys!(sys::PSY.System, name::String)
    device = PSY.get_component(PSY.DynamicInverter, sys, name)
    bus = PSY.get_number(PSY.get_bus(PSY.get_component(PSY.StaticInjection, sys, name)))

    sim = PSID.Simulation!(
        PSID.MassMatrixModel,
        sys,
        pwd(),
        (0.0, 1.0),
        console_level = PSID_CONSOLE_LEVEL,
        file_level = PSID_FILE_LEVEL,
    )

    @warn PSID.read_initial_conditions(sim)
    x₀_dict = PSID.read_initial_conditions(sim)[PSY.get_name(device)]
    x₀ = [value for (key, value) in x₀_dict]

    setpoints = PSID.get_setpoints(sim)["gen1"]
    refs = [setpoints["V_ref"], setpoints["ω_ref"], setpoints["P_ref"], setpoints["Q_ref"]]
    Vr0 = PSID.read_initial_conditions(sim)["V_R"][bus]
    Vi0 = PSID.read_initial_conditions(sim)["V_I"][bus]
    return x₀, refs, Vr0, Vi0
end

function set_bus_from_source(available_source::PSY.Source)
    Vsource = PSY.get_internal_voltage(available_source)
    PSY.set_magnitude!(PSY.get_bus(available_source), Vsource)
    θsource = PSY.get_internal_angle(available_source)
    PSY.set_angle!(PSY.get_bus(available_source), θsource)
end

function get_total_current_series(sim)
    ir_total = []
    ii_total = []
    for (i, g) in enumerate(
        PSY.get_components(
            PSY.DynamicInjection,
            sim.sys,
            x -> typeof(x) !== PSY.PeriodicVariableSource,
        ),
    )
        results = PSID.read_results(sim)
        if i == 1
            ir_total = PSID.get_real_current_series(results, PSY.get_name(g))[2]
            ii_total = PSID.get_imaginary_current_series(results, PSY.get_name(g))[2]
        else
            ir_total .+= PSID.get_real_current_series(results, PSY.get_name(g))[2]
            ii_total .+= PSID.get_imaginary_current_series(results, PSY.get_name(g))[2]
        end
    end
    data_array = zeros(Float64, (2, length(ir_total)))
    data_array[1, :] .= ir_total
    data_array[2, :] .= ii_total
    return data_array
end

function build_train_system(
    sys_surr_original::PSY.System,
    sys_pvs_original::PSY.System,
    surrogate_area_name::String,
)
    sys_surr = deepcopy(sys_surr_original)
    sys_pvs = deepcopy(sys_pvs_original)
    non_surrogate_buses = collect(
        PSY.get_components(
            PSY.Bus,
            sys_surr,
            x -> PSY.get_name(PSY.get_area(x)) != surrogate_area_name,
        ),
    )
    if length(non_surrogate_buses) != 1
        @error "Must have one-non surrogate bus designated in surrogate system to add the PVS to"
        return
    end
    non_surrogate_bus = non_surrogate_buses[1]
    PSY.set_bustype!(non_surrogate_bus, PSY.BusTypes.REF)

    sources = PSY.get_components(PSY.Source, sys_pvs)
    for s in sources
        pvs = PSY.get_dynamic_injector(s)
        PSY.remove_component!(sys_pvs, pvs)
        PSY.remove_component!(sys_pvs, s)

        PSY.set_bus!(s, non_surrogate_bus)
        PSY.add_component!(sys_surr, s)
        PSY.add_component!(sys_surr, pvs, s)
    end
    return sys_surr
end

#Label an collection of buses with a name
function label_area!(sys::PSY.System, bus_numbers, area_name::String)
    buses = collect(PSY.get_components(PSY.Bus, sys))
    areas = collect(PSY.get_components(PSY.Area, sys))
    for area in areas
        if PSY.get_name(area) == area_name
            @error "area already exists"
            return 0
        end
    end
    surrogate_area = PSY.Area(; name = area_name)
    PSY.add_component!(sys, surrogate_area)
    for bus in buses
        if PSY.get_number(bus) in bus_numbers
            PSY.set_area!(bus, surrogate_area)
        end
    end
end

function check_single_connecting_line_condition(sys::PSY.System)
    areas = PSY.get_components(PSY.Area, sys)
    if length(areas) != 2
        @warn "There aren't two areas in this system. Cannot check single line condition"
    end
    branches = collect(PSY.get_components(PSY.Branch, sys))
    count_connections = 0
    for branch in branches
        area_from = PSY.get_name(PSY.get_area(PSY.get_from(PSY.get_arc(branch))))
        area_to = PSY.get_name(PSY.get_area(PSY.get_to(PSY.get_arc(branch))))
        if (area_from != area_to)
            count_connections += 1
        end
    end
    if count_connections == 1
        return true
    else
        @warn "Not exactly one connections between the two areas"
        return false
    end
end

function get_parameters(inv::PSY.DynamicInverter)
    #pll
    p = Vector{Float64}(undef, 23)
    p[1] = PSY.get_ω_lp(inv.freq_estimator)
    p[2] = PSY.get_kp_pll(inv.freq_estimator)
    p[3] = PSY.get_ki_pll(inv.freq_estimator)
    #outer control
    p[4] = PSY.get_Ta(inv.outer_control.active_power)
    p[5] = PSY.get_kd(inv.outer_control.active_power)
    p[6] = PSY.get_kω(inv.outer_control.active_power)
    p[7] = PSY.get_kq(inv.outer_control.reactive_power)
    p[8] = PSY.get_ωf(inv.outer_control.reactive_power)
    #inner control
    p[9] = PSY.get_kpv(inv.inner_control)
    p[10] = PSY.get_kiv(inv.inner_control)
    p[11] = PSY.get_kffv(inv.inner_control)
    p[12] = PSY.get_rv(inv.inner_control)
    p[13] = PSY.get_lv(inv.inner_control)
    p[14] = PSY.get_kpc(inv.inner_control)
    p[15] = PSY.get_kic(inv.inner_control)
    p[16] = PSY.get_kffi(inv.inner_control)
    p[17] = PSY.get_ωad(inv.inner_control)
    p[18] = PSY.get_kad(inv.inner_control)
    #lcl
    p[19] = PSY.get_lf(inv.filter)
    p[20] = PSY.get_rf(inv.filter)
    p[21] = PSY.get_cf(inv.filter)
    p[22] = PSY.get_lg(inv.filter)
    p[23] = PSY.get_rg(inv.filter)
    return p
end

function remove_area(sys_original::PSY.System, area_name::String)
    sys = deepcopy(sys_original)
    (
        length(
            collect(PSY.get_components(PSY.Area, sys, x -> PSY.get_name(x) == area_name)),
        ) == 1
    ) || @warn "area with name not found or multiple areas with same name"
    connecting_bus_name = nothing

    static_injectors = collect(
        PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.StaticInjection),
    )
    for static_injector in static_injectors
        if PSY.get_name(PSY.get_area(PSY.get_bus(static_injector))) == area_name
            dynamic_injector = PSY.get_dynamic_injector(static_injector)
            (dynamic_injector !== nothing) && PSY.remove_component!(sys, dynamic_injector)
            PSY.remove_component!(sys, static_injector)
        end
    end

    branches = collect(PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.Branch))
    for branch in branches     #DOES ORDER MATTER ?  
        area_name_from = PSY.get_name(PSY.get_area(PSY.get_from(PSY.get_arc(branch))))
        area_name_to = PSY.get_name(PSY.get_area(PSY.get_to(PSY.get_arc(branch))))
        if (area_name_from == area_name_to == area_name)
            arc = PSY.get_arc(branch)
            PSY.remove_component!(sys, arc)
            PSY.remove_component!(sys, branch)
        end
        if (area_name_from != area_name_to)
            if (area_name_from == area_name)
                connecting_bus_name = PSY.get_name(PSY.get_from(PSY.get_arc(branch)))
            else
                connecting_bus_name = PSY.get_name(PSY.get_to(PSY.get_arc(branch)))
            end
        end
    end

    buses = collect(PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.Bus))
    for bus in buses
        if (PSY.get_name(PSY.get_area(bus)) == area_name) &&
           (PSY.get_name(bus) != connecting_bus_name)
            PSY.remove_component!(sys, bus)
        end
    end
    return sys
end
