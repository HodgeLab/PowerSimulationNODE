
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

function create_surrogate_training_system(sys_original, surrogate_area_name, pvs_data)
    sys = deepcopy(sys_original)
    (
        length(
            collect(
                PSY.get_components(
                    PSY.Area,
                    sys,
                    x -> PSY.get_name(x) == surrogate_area_name,
                ),
            ),
        ) == 1
    ) || @warn "area with name not found or multiple areas with same name"
    connecting_bus_name = nothing

    static_injectors = collect(
        PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.StaticInjection),
    )
    for static_injector in static_injectors
        if PSY.get_name(PSY.get_area(PSY.get_bus(static_injector))) != surrogate_area_name
            dynamic_injector = PSY.get_dynamic_injector(static_injector)
            (dynamic_injector !== nothing) && PSY.remove_component!(sys, dynamic_injector)
            PSY.remove_component!(sys, static_injector)
        end
    end

    connecting_bus_names = []
    branches = collect(PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.Branch))
    for branch in branches
        area_name_from = PSY.get_name(PSY.get_area(PSY.get_from(PSY.get_arc(branch))))
        area_name_to = PSY.get_name(PSY.get_area(PSY.get_to(PSY.get_arc(branch))))
        if (area_name_from == area_name_to) && (area_name_from != surrogate_area_name)
            arc = PSY.get_arc(branch)
            PSY.remove_component!(sys, arc)
            PSY.remove_component!(sys, branch)
        end
        if (area_name_from != area_name_to)
            if (area_name_from != surrogate_area_name)
                connecting_bus_name = PSY.get_name(PSY.get_from(PSY.get_arc(branch)))
                push!(connecting_bus_names, connecting_bus_name)
            else
                connecting_bus_name = PSY.get_name(PSY.get_to(PSY.get_arc(branch)))
                push!(connecting_bus_names, connecting_bus_name)
            end
        end
    end

    buses = collect(PSY.get_components(PSY.Component, sys, x -> typeof(x) <: PSY.Bus))
    for bus in buses
        if (PSY.get_name(PSY.get_area(bus)) != surrogate_area_name) &&
           (PSY.get_name(bus) ∉ connecting_bus_names)
            PSY.remove_component!(sys, bus)
        end
    end

    connecting_branches = find_connecting_branches(sys, surrogate_area_name)
    for (fault_id, PVSDatas) in pvs_data
        for PVSData in PVSDatas
            branch = PSY.get_components_by_name(PSY.ACBranch, sys, PVSData.branch_name)
            @assert length(branch) == 1
            branch = branch[1]
            branch_name = PSY.get_name(branch)
            if PVSData.from_or_to == "from"
                bus_to_add_pvs = PSY.get_from(PSY.get_arc(branch))
            elseif PVSData.from_or_to == "to"
                bus_to_add_pvs = PSY.get_to(PSY.get_arc(branch))
            else
                @error "invalid value of from_or_to"
            end
            @warn bus_to_add_pvs
            pvs_name = string(fault_id, "_", branch_name)

            PSY.set_bustype!(bus_to_add_pvs, PSY.BusTypes.REF)
            source = PSY.Source(
                name = pvs_name,
                active_power = PVSData.P0,
                available = false,
                reactive_power = PVSData.Q0,
                bus = bus_to_add_pvs,
                R_th = 1e-6, #0.0
                X_th = 1e-6, #5e-6
                internal_voltage = PVSData.V0,
                internal_angle = PVSData.θ0,
            )
            @warn "Voltage Bias when building PVS", PVSData.internal_voltage_bias
            pvs = PSY.PeriodicVariableSource(
                name = PSY.get_name(source),
                R_th = PSY.get_R_th(source),
                X_th = PSY.get_X_th(source),
                internal_voltage_bias = PVSData.internal_voltage_bias,
                internal_voltage_frequencies = PVSData.internal_voltage_frequencies,
                internal_voltage_coefficients = PVSData.internal_voltage_coefficients,
                internal_angle_bias = PVSData.internal_angle_bias,
                internal_angle_frequencies = PVSData.internal_angle_frequencies,
                internal_angle_coefficients = PVSData.internal_angle_coefficients,
            )
            PSY.add_component!(sys, source)
            PSY.add_component!(sys, pvs, source)
        end
    end
    return sys
end

function pad_tanh(t, y)
    first = y[1]
    last = y[end]
    Δt = t[2] - t[1] #assume linear spacing
    n = length(t)
    A = (y[1] - y[end]) / 2
    C = y[end] + A

    B = 20 / (t[end] - t[1])
    @info A, B, C
    @info y[end]
    @info y[1]
    t_add = (t[end] + Δt):Δt:(t[end] + (t[end] - t[1]))
    y_add = A * tanh.((t_add .- t_add[Int(length(t_add) / 2)]) .* B) .+ C
    t_new = vcat(t, t_add)
    y_new = vcat(y, y_add)
    return (t_new, y_new)
end

function find_connecting_branches(sys, area_name)
    connecting_branches = []
    ac_branches = PSY.get_components(PSY.Device, sys, x -> typeof(x) <: PSY.ACBranch)
    for b in ac_branches
        from_bus = PSY.get_from(PSY.get_arc(b))
        to_bus = PSY.get_to(PSY.get_arc(b))
        if (PSY.get_name(PSY.get_area(from_bus)) == area_name) &&
           (PSY.get_name(PSY.get_area(to_bus)) != area_name)
            push!(connecting_branches, b)
        end
        if (PSY.get_name(PSY.get_area(to_bus)) == area_name) &&
           (PSY.get_name(PSY.get_area(from_bus)) != area_name)
            push!(connecting_branches, b)
        end
    end
    return connecting_branches
end

function all_line_trips(sys, t_fault)
    perturbations = []
    lines = PSY.get_components(PSY.Line, sys)
    for l in lines
        push!(
            perturbations,
            PowerSimulationsDynamics.BranchTrip(t_fault, PSY.Line, PSY.get_name(l)),
        )
    end
    return perturbations
end
