function Base.show(io::IO, ::MIME"text/plain", params::NODETrainParams)
    for field_name in fieldnames(NODETrainParams)
        if field_name == :training_groups
            println(io, "$field_name =")
            for v in getfield(params, field_name)
                println(io, "\ttspan: ", v.tspan)
                println(io, "\tmultiple shoot group size: ", v.multiple_shoot_group_size)
                println(
                    io,
                    "\tmultiple_shoot_continuity_term: ",
                    v.multiple_shoot_continuity_term,
                )
                println(io, "\tbatching factor: ", v.batching_sample_factor)
            end
        else
            println(io, "$field_name = ", getfield(params, field_name))
        end
    end
end

function visualize_training(params::NODETrainParams; visualize_level = 1)
    @debug dump(params)
    path_to_input = params.input_data_path
    path_to_output = joinpath(params.output_data_path, params.train_id)

    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    println("--------------------------------")
    println("TRAIN ID: ", params.train_id)
    println("TOTAL TIME: ", output_dict["total_time"])
    println("TOTAL ITERATIONS: ", output_dict["total_iterations"])
    println("FINAL LOSS: ", output_dict["final_loss"])
    println("--------------------------------")

    if params.output_mode == 2
        plots = visualize_2(params, path_to_output, path_to_input)
        for p in plots
            png(p, path_to_output)
        end
        return
    elseif params.output_mode == 3
        plots = visualize_3(params, path_to_output, path_to_input, visualize_level)
        for (i, p) in enumerate(plots)
            Plots.png(p, joinpath(path_to_output, string("plot_", i)))
        end
        return
    end
end

function visualize_2(params, path_to_output, path_to_input)
    df_loss = DataFrames.DataFrame(Arrow.Table(joinpath(path_to_output, "loss")))
    p1 = Plots.plot(df_loss.Loss, title = "Loss")
    p2 = Plots.plot(df_loss.RangeCount, title = "Range Count")
    return Plots.plot(p1, p2, layout = (2, 1))
end

function visualize_3(params, path_to_output, path_to_input, visualize_level)
    df_loss = DataFrames.DataFrame(Arrow.Table(joinpath(path_to_output, "loss")))
    list_plots = []
    p1 = Plots.plot(df_loss.Loss, title = "Loss")
    p2 = Plots.plot(df_loss.RangeCount, title = "Range Count")
    p = Plots.plot(p1, p2, layout = (2, 1))
    push!(list_plots, p)

    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    PVS_name = df_loss.PVS_name[:]
    if visualize_level == 1
        transition_indices = [length(PVS_name)]   #LEVEL 1: plot end result only (the result used to calculate final loss)
    elseif visualize_level == 2
        transition_indices = find_transition_indices(PVS_name)  #LEVEL 2: plot when moving to new fault(s)
    elseif visualize_level == 3
        transition_indices = find_transition_indices(df_loss.RangeCount)  #LEVEL 3: plot when moving to new data range
    elseif visualize_level == 4
        transition_indices = collect(1:length(PVS_name)) #LEVEL 4: plot every iteration moving to new data range
    else
        @warn "Invalid value for parameter visualize_level"
    end
    df_predictions =
        DataFrames.DataFrame(Arrow.Table(joinpath(path_to_output, "predictions")))
    TrainInputs = Serialization.deserialize(joinpath(params.input_data_path, "data"))
    tsteps = TrainInputs.tsteps
    fault_data = TrainInputs.fault_data

    for i in transition_indices
        ir_preds = df_predictions[i, "ir_prediction"]
        ii_preds = df_predictions[i, "ii_prediction"]
        t_preds = df_predictions[i, "t_prediction"]
        i_true = concatonate_i_true(fault_data, df_loss[i, :PVS_name], :)
        ir_true = i_true[1, :]
        ii_true = i_true[2, :]
        t_all = concatonate_t(tsteps, df_loss[i, :PVS_name], :)
        p3 = Plots.scatter(t_all', ir_true, ms = 2, msw = 0, label = "truth")
        p4 = Plots.scatter(t_all', ii_true, ms = 2, msw = 0, label = "truth")

        for (i, t_pred) in enumerate(t_preds)
            Plots.scatter!(p3, t_preds[i], ir_preds[i], ms = 2, msw = 0, legend = false)
            Plots.scatter!(p4, t_preds[i], ii_preds[i], ms = 2, msw = 0, legend = false)
        end
        p = Plots.plot(
            p3,
            p4,
            title = string(df_loss[i, :PVS_name], " loss: ", output_dict["final_loss"]),
            layout = (2, 1),
        )
        push!(list_plots, p)
    end
    return list_plots
end

function find_transition_indices(list)
    transition_indices = Int[]

    for i in 1:(length(list) - 1)
        if list[i] != list[i + 1]
            push!(transition_indices, i)
        end
    end
    push!(transition_indices, length(list))
    return transition_indices
end

function visualize_summary(output_data_path)
    output_directories = readdir(output_data_path)
    high_level_outputs_dict = Dict{String, Dict{String, Any}}()
    for dir in output_directories
        output_dict = JSON3.read(
            read(joinpath(output_data_path, dir, "high_level_outputs")),
            Dict{String, Any},
        )
        high_level_outputs_dict[output_dict["train_id"]] = output_dict
    end
    p = Plots.scatter()
    for (key, value) in high_level_outputs_dict
        Plots.scatter!(
            p,
            (value["total_time"], value["final_loss"]),
            label = value["train_id"],
            xlabel = "total time (s)",
            ylabel = "final loss",
            markersize = 3,
            markerstrokewidth = 0,
        )
        Plots.annotate!(
            value["total_time"],
            value["final_loss"],
            Plots.text(value["train_id"], :red, 8),
        )
    end
    return p
end

function plot_pvs(tsteps, pvs::PSY.PeriodicVariableSource, xaxis)
    V = zeros(length(tsteps))
    V = V .+ PSY.get_internal_voltage_bias(pvs)
    retrieved_freqs = PSY.get_internal_voltage_frequencies(pvs)
    coeffs = PSY.get_internal_voltage_coefficients(pvs)
    for (i, ω) in enumerate(retrieved_freqs)
        V += coeffs[i][1] * sin.(ω .* tsteps)
        V += coeffs[i][2] * cos.(ω .* tsteps)
    end

    θ = zeros(length(tsteps))
    θ = θ .+ PSY.get_internal_angle_bias(pvs)
    retrieved_freqs = PSY.get_internal_angle_frequencies(pvs)
    coeffs = PSY.get_internal_angle_coefficients(pvs)
    for (i, ω) in enumerate(retrieved_freqs)
        θ += coeffs[i][1] * sin.(ω .* tsteps)
        θ += coeffs[i][2] * cos.(ω .* tsteps)
    end
    return tsteps, V, θ
end

function print_train_parameter_overview(input_folder)
    Matrix = Any[]
    header = Symbol[]
    files = filter(x -> contains(x, ".json"), readdir(input_folder, join = true))   #TODO, make clean
    files = filter(x -> !contains(x, "data"), files)
    files = filter(x -> !contains(x, "system"), files)
    files = filter(x -> !contains(x, "sample"), files)
    @warn files
    for (i, f) in enumerate(files)
        Matrix_row = Any[]
        params = NODETrainParams(f)
        for fieldname in fieldnames(NODETrainParams)
            exclude_fields = [
                :optimizer_adjust,
                :optimizer_adjust_η,
                :base_path,
                :output_mode,
                :input_data_path,
                :output_data_path,
                :verify_psid_node_off,
                :graphical_report_mode,
            ]
            if !(fieldname in exclude_fields)
                push!(Matrix_row, getfield(params, fieldname))
                if i == 1
                    push!(header, fieldname)
                end
            end
        end
        Matrix_row = reshape(Matrix_row, 1, :)
        if i == 1
            Matrix = Matrix_row
        else
            Matrix = vcat(Matrix, Matrix_row)
        end
    end

    common_params_indices = [all(x -> x == col[1], col) for col in eachcol(Matrix)]
    changing_params_indices = [!(all(x -> x == col[1], col)) for col in eachcol(Matrix)]

    common_params = Matrix[1:1, common_params_indices]
    common_header = header[common_params_indices]
    changing_params = Matrix[:, changing_params_indices]
    changing_header = header[changing_params_indices]

    print("COMMON PARAMETERS:\n")
    PrettyTables.pretty_table(common_params, header = common_header)
    print("CHANGING PARAMETERS:\n")
    PrettyTables.pretty_table(
        changing_params,
        header = changing_header,
        highlighters = (Highlighter(
            (data, i, j) -> true,
            Crayon(bold = true, background = :red),
        )),
    )
end
