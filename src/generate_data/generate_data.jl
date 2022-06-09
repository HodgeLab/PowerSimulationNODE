
"""
    mutable struct GenerateDataParams

# Fields
- `solver::String`: solver used for the generating ground truth data. Valid Values ["Rodas4"]
- `solver_tols:: Tuple{Float64, Float64}`: solver tolerances (abstol, reltol).
- `tspan::Tuple{Float64, Float64}`: timespan of ground truth data.
- `steps::Int64`: Total number of steps for a single fault in ground truth data. The distribution of steps is determined by `tsteps_spacing`.
- `tsteps_spacing:: ["linear"]`: Determines the distribution of save data points. 
- `base_path::String`: Project directory
- `output_data_path::String`: Relative path from project directory to where outputs are written (inputs to training)
"""
mutable struct GenerateDataParams
    solver::String
    solver_tols::Tuple{Float64, Float64}
    tspan::Tuple{Float64, Float64}
    steps::Int64
    tsteps_spacing::String
    formulation::String
end

function GenerateDataParams(;
    solver = "Rodas4",
    solver_tols = (1e-6, 1e-6),
    tspan = (0.0, 1.0),
    steps = 100,
    tsteps_spacing = "linear",
    formulation = "MassMatrix",
)
    GenerateDataParams(solver, solver_tols, tspan, steps, tsteps_spacing, formulation)
end

function GenerateDataParams(file::AbstractString)
    return JSON3.read(read(file), GenerateDataParams)
end

function generate_pvs_data(sys_full, pvs_coeffs, surrogate_area_name)
    connecting_branches = find_connecting_branches(sys_full, surrogate_area_name)
    pvs_data = Dict{Int, Array{PVSData}}()
    for (k, v) in pvs_coeffs
        bus_results = PowerFlows.run_powerflow(sys_full)["bus_results"]
        flow_results = PowerFlows.run_powerflow(sys_full)["flow_results"]

        PVSDatas = PVSData[]
        @warn typeof(PVSDatas)
        for (i, branch) in enumerate(connecting_branches)
            from_bus = PSY.get_from(PSY.get_arc(branch))
            to_bus = PSY.get_to(PSY.get_arc(branch))
            @assert PSY.get_area(from_bus) != PSY.get_area(to_bus)
            if PSY.get_area(from_bus) == surrogate_area_name
                from_or_to = "to"
            else
                from_or_to = "from"
            end
            branch_name = PSY.get_name(branch)
            from_bus = PSY.get_from(PSY.get_arc(branch))
            to_bus = PSY.get_from(PSY.get_arc(branch))
            display(bus_results)
            display(flow_results)

            if from_or_to == "from"
                pvs_bus_results =
                    filter(row -> row.bus_number == PSY.get_number(from_bus), bus_results)
                V0 = pvs_bus_results.Vm[1]
                θ0 = pvs_bus_results.θ[1]
                connecting_branch_results =
                    filter(row -> row.line_name == branch_name, flow_results)
                P0 = connecting_branch_results.P_from_to[1]
                Q0 = connecting_branch_results.Q_from_to[1]
            elseif from_or_to == "to"
                pvs_bus_results =
                    filter(row -> row.bus_number == PSY.get_number(to_bus), bus_results)
                V0 = pvs_bus_results.Vm[1]
                θ0 = pvs_bus_results.θ[1]
                connecting_branch_results =
                    filter(row -> row.line_name == branch_name, flow_results)
                P0 = connecting_branch_results.P_to_from[1]
                Q0 = connecting_branch_results.Q_to_from[1]
            end

            current_pvs_data = PVSData(
                0.0,
                pvs_coeffs[k][i].internal_voltage_frequencies,
                pvs_coeffs[k][i].internal_voltage_coefficients,
                0.0,
                pvs_coeffs[k][i].internal_angle_frequencies,
                pvs_coeffs[k][i].internal_angle_coefficients,
                V0,
                θ0,
                P0,
                Q0,
                PSY.get_name(branch),
                from_or_to,
            )
            add_calculated_bias!(current_pvs_data)
            @warn current_pvs_data
            push!(PVSDatas, current_pvs_data)
        end
        pvs_data[k] = PVSDatas
    end
    return pvs_data
end
function add_calculated_bias!(x::PVSData)
    V = x.V0
    θ = x.θ0
    for c in x.internal_voltage_coefficients
        V -= c[2]  #cosine part 
    end
    for c in x.internal_angle_coefficients
        θ -= c[2]  #cosine part 
    end
    x.internal_voltage_bias = V
    x.internal_angle_bias = θ
end

#Method for generating PVS data from fault data
function generate_pvs_data(sys_full, faults, params, surrogate_area_name; pad_signal = true)
    connecting_branches = find_connecting_branches(sys_full, surrogate_area_name)
    pvs_data = Dict{Int, Array{PVSData}}()
    for (i, fault) in enumerate(faults)
        solver = instantiate_solver(params)
        solver_tols = params.solver_tols
        tspan = params.tspan
        steps = params.steps
        tsteps = tspan[1]:((tspan[2] - tspan[1]) / steps):tspan[2]

        bus_results = PowerFlows.run_powerflow(sys_full)["bus_results"]
        flow_results = PowerFlows.run_powerflow(sys_full)["flow_results"]

        sim = PSID.Simulation!(
            PSID.MassMatrixModel,
            sys_full,
            pwd(),
            tspan,
            fault,
            console_level = PSID_CONSOLE_LEVEL,
            file_level = PSID_FILE_LEVEL,
        )

        PSID.execute!(
            sim,
            solver,
            reltol = 1e-3,# solver_tols[1],
            abstol = 1e-3, # solver_tols[2],
            dtmax = 0.02,
            reset_simulation = false,
            saveat = tsteps,
            enable_progress_bar = false,
        )
        results = PSID.read_results(sim)

        PVSDatas = PVSData[]
        @warn typeof(PVSDatas)
        for branch in connecting_branches
            from_bus = PSY.get_from(PSY.get_arc(branch))
            to_bus = PSY.get_to(PSY.get_arc(branch))
            @assert PSY.get_area(from_bus) != PSY.get_area(to_bus)
            if PSY.get_area(from_bus) == surrogate_area_name
                from_or_to = "to"
            else
                from_or_to = "from"
            end
            branch_name = PSY.get_name(branch)
            from_bus = PSY.get_from(PSY.get_arc(branch))
            to_bus = PSY.get_from(PSY.get_arc(branch))

            if from_or_to == "from"
                Vm = PSID.get_voltage_magnitude_series(results, PSY.get_number(from_bus))[2]
                Vθ = PSID.get_voltage_angle_series(results, PSY.get_number(from_bus))[2]
                connecting_branch_results =
                    filter(row -> row.line_name == branch_name, flow_results)
                P0 = connecting_branch_results.P_from_to[1]
                Q0 = connecting_branch_results.Q_from_to[1]
                t = collect(tsteps)
            elseif from_or_to == "to"
                Vm = PSID.get_voltage_magnitude_series(results, PSY.get_number(to_bus))[2]
                Vθ = PSID.get_voltage_angle_series(results, PSY.get_number(to_bus))[2]
                connecting_branch_results =
                    filter(row -> row.line_name == branch_name, flow_results)
                P0 = connecting_branch_results.P_to_from[1]
                Q0 = connecting_branch_results.Q_to_from[1]
                t = collect(tsteps)
            end

            if pad_signal
                t_new, Vm_new = pad_tanh(t, Vm)
                t_new, Vθ_new = pad_tanh(t, Vθ)
                t = t_new
                Vm = Vm_new
                Vθ = Vθ_new
            end
            #Dft 
            N = length(t)
            fs = (N - 1) / (t[end] - t[1])
            freqs = FFTW.fftfreq(N, fs)
            freqs_pos = freqs[freqs .>= 0] * (2 * pi)

            F_V = FFTW.fft(Vm)
            F_V = F_V[freqs .>= 0]
            F_V = F_V / N
            F_V[2:end] = F_V[2:end] * 2
            internal_voltage_bias = real(F_V[1])
            internal_voltage_frequencies = freqs_pos[2:end]
            internal_voltage_coefficients = [(-imag(f), real(f)) for f in F_V[2:end]]

            F_θ = FFTW.fft(Vθ)
            F_θ = F_θ[freqs .>= 0]
            F_θ = F_θ / N
            F_θ[2:end] = F_θ[2:end] * 2
            internal_angle_bias = real(F_θ[1])
            internal_angle_frequencies = freqs_pos[2:end]
            internal_angle_coefficients = [(-imag(f), real(f)) for f in F_θ[2:end]]
            @warn "Initial voltage magnitude", Vm[1]
            @warn "Initial voltage angle", Vθ[1]
            @warn "voltage magnitude bias ", internal_voltage_bias
            @warn "voltage angle bias ", internal_angle_bias
            V0 = Vm[1]
            θ0 = Vθ[1]
            push!(
                PVSDatas,
                PVSData(
                    internal_voltage_bias,
                    internal_voltage_frequencies,
                    internal_voltage_coefficients,
                    internal_angle_bias,
                    internal_angle_frequencies,
                    internal_angle_coefficients,
                    V0,
                    θ0,
                    P0,
                    Q0,
                    PSY.get_name(branch),
                    from_or_to,
                ),
            )
        end
        pvs_data[i] = PVSDatas
    end
    return pvs_data
end

function _add_to_fault_data_dict(fault_data, value, timeseries)
    if haskey(fault_data, value.type)
        if haskey(fault_data[value.type], value.name)
            if haskey(fault_data[value.type][value.name], value.state)
                @error "Dictionary already contains the data you are trying to add"
            else
                fault_data[value.type][value.name][value.state] = timeseries
            end
        else
            fault_data[value.name] =
                Dict{Symbol, Vector{Float64}}(value.state => timeseries)
        end
    else
        fault_data[value.type] = Dict{String, Dict{Symbol, Vector{Float64}}}(
            value.name => Dict(value.state => timeseries),
        )
    end
end

"""
"""
function generate_train_data(sys_train::PSY.System, params::GenerateDataParams)
    tspan = params.tspan
    steps = params.steps
    if params.tsteps_spacing == "linear"
        tsteps = tspan[1]:((tspan[2] - tspan[1]) / steps):tspan[2]
    end
    solver = instantiate_solver(params)
    abstol = params.solver_tols[2]
    reltol = params.solver_tols[1]

    sources = collect(PSY.get_components(PSY.Source, sys_train))
    buses_with_sources = PSY.get_bus.(sources)
    unique_buses_with_sources = unique(buses_with_sources)
    sources_at_each_bus =
        [count(==(element), buses_with_sources) for element in unique_buses_with_sources]
    @assert all(x -> x == sources_at_each_bus[1], sources_at_each_bus)  #all buses with sources should have the same number of sources 
    sources_per_bus = sources_at_each_bus[1]

    #find connecting branches
    connecting_branches = String[]
    for b in PSY.get_components(PSY.ACBranch, sys_train)
        to_bus = PSY.get_to(PSY.get_arc(b))
        from_bus = PSY.get_from(PSY.get_arc(b))
        if (from_bus in unique_buses_with_sources) || (to_bus in unique_buses_with_sources)
            push!(connecting_branches, PSY.get_name(b))
        end
    end

    train_data = TrainData[]
    for source_index in 1:sources_per_bus
        available_sources = _set_available_source_by_name_index!(
            sys_train,
            unique_buses_with_sources,
            source_index,
        )
        set_bus_from_source.(available_sources)  #Bus voltage is used in power flow. Need to set bus voltage from soure internal voltage of source

        sim_full = PSID.Simulation!(
            PSID.MassMatrixModel,
            sys_train,
            pwd(),
            tspan,
            console_level = PSID_CONSOLE_LEVEL,
            file_level = PSID_FILE_LEVEL,
        )
        display(sim_full)

        PSID.execute!(
            sim_full,
            solver,
            abstol = abstol,
            reltol = reltol,
            reset_simulation = false,
            saveat = tsteps,
            enable_progress_bar = false,
        )

        ground_truth_current = zeros(2 * length(connecting_branches), length(tsteps))
        connecting_impedance = zeros(length(connecting_branches), 2)
        powerflow = zeros(length(connecting_branches) * 4)

        for (i, branch_name) in enumerate(connecting_branches)
            ground_truth_current[2 * i - 1, :] = get_total_current_series(sim_full)[1, :]
            ground_truth_current[2 * i, :] = get_total_current_series(sim_full)[2, :]
            #ground_truth_ir[i, :] = get_branch_current(branch_name, :Ir) #TODO:  Get branch current instead when this issue closes: https://github.com/NREL-SIIP/PowerSimulationsDynamics.jl/issues/224
            #ground_truth_ii[i, :] = get_branch_current(branch_name, :Ii)
            connecting_impedance[i, :] =
                _get_branch_plus_source_impedance(sys_train, branch_name)
            powerflow[(i * 4 - 3):(i * 4)] =
                _get_powerflow_opposite_source(sys_train, branch_name)
        end
        @warn powerflow
        push!(
            train_data,
            TrainData(tsteps, ground_truth_current, connecting_impedance, powerflow),
        )
    end

    return SurrogateTrainInputs(connecting_branches, train_data)
end

function _get_branch_plus_source_impedance(sys_train, branch_name)
    @assert length(PSY.get_components_by_name(PSY.ACBranch, sys_train, branch_name)) == 1
    ac_branch = PSY.get_components_by_name(PSY.ACBranch, sys_train, branch_name)[1]
    bus_from = PSY.get_from(PSY.get_arc(ac_branch))
    bus_to = PSY.get_to(PSY.get_arc(ac_branch))
    source_active = collect(
        PSY.get_components(
            PSY.Source,
            sys_train,
            x -> PSY.get_available(x) && PSY.get_bus(x) in [bus_from, bus_to],
        ),
    )
    @assert length(source_active) == 1
    source_active = source_active[1]
    return [
        PSY.get_x(ac_branch) + PSY.get_X_th(source_active),
        PSY.get_r(ac_branch) + PSY.get_R_th(source_active),
    ]
end

function _get_powerflow_opposite_source(sys_train, branch_name)
    flow_results = PowerFlows.run_powerflow(sys_train)["flow_results"]
    bus_results = PowerFlows.run_powerflow(sys_train)["bus_results"]
    display(flow_results)
    display(bus_results)

    connecting_branch_results = filter(row -> row.line_name == branch_name, flow_results)
    @assert size(connecting_branch_results)[1] == 1

    bus_from = connecting_branch_results.bus_from[1]
    bus_to = connecting_branch_results.bus_to[1]

    source_active = collect(
        PSY.get_components(
            PSY.Source,
            sys_train,
            x ->
                PSY.get_available(x) &&
                    PSY.get_number(PSY.get_bus(x)) in [bus_from, bus_to],
        ),
    )
    @assert length(source_active) == 1
    source_active = source_active[1]

    if PSY.get_number(PSY.get_bus(source_active)) == bus_from
        P_pf = connecting_branch_results.P_to_from[1] / PSY.get_base_power(sys_train)
        Q_pf = connecting_branch_results.Q_to_from[1] / PSY.get_base_power(sys_train)
        opposite_source_bus_results = filter(row -> row.bus_number == bus_to, bus_results)
        V_pf = opposite_source_bus_results.Vm[1]
        θ_pf = opposite_source_bus_results.θ[1]
        return [P_pf, Q_pf, V_pf, θ_pf]
    elseif PSY.get_number(PSY.get_bus(source_active)) == bus_to
        P_pf = connecting_branch_results.P_from_to[1] / PSY.get_base_power(sys_train)
        Q_pf = connecting_branch_results.Q_from_to[1] / PSY.get_base_power(sys_train)
        opposite_source_bus_results = filter(row -> row.bus_number == bus_from, bus_results)
        V_pf = opposite_source_bus_results.Vm[1]
        θ_pf = opposite_source_bus_results.θ[1]
        return [P_pf, Q_pf, V_pf, θ_pf]
    else
        @error "Didn't find a source at the correct bus"
    end
end

function _set_available_source_by_name_index!(
    sys_train,
    unique_buses_with_sources,
    source_index,
)
    available_sources = []
    for b in unique_buses_with_sources
        sources_at_bus =
            collect(PSY.get_components(PSY.Source, sys_train, x -> PSY.get_bus(x) == b))
        found_source = false
        for s in sources_at_bus
            name = PSY.get_name(s)
            x = parse(Int, split(name, "_")[1])
            if source_index == x
                PSY.set_available!(s, true)
                push!(available_sources, s)
                found_source = true
            else
                PSY.set_available!(s, false)
            end
        end
        if !found_source
            @error "Didn't find a source with appropriate index at bus"
            return
        end
    end
    return available_sources
end

function set_bus_from_source(available_source::PSY.Source)
    Vsource = PSY.get_internal_voltage(available_source)
    PSY.set_magnitude!(PSY.get_bus(available_source), Vsource)
    θsource = PSY.get_internal_angle(available_source)
    PSY.set_angle!(PSY.get_bus(available_source), θsource)
end

#TODO:  Get branch current instead when this issue closes: https://github.com/NREL-SIIP/PowerSimulationsDynamics.jl/issues/224
#Then can delete this function.
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
