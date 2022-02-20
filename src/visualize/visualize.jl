function Base.show(io::IO, ::MIME"text/plain", params::NODETrainParams)
    for field_name in fieldnames(NODETrainParams)
        if field_name == :training_groups
            println(io, "$field_name =")
            for v in getfield(params, field_name)
                println(io, "\ttspan: ", v.tspan)
                println(io, "\tshoot times: ", v.shoot_times)
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

function visualize_training(input_params_file::String; visualize_level = 1)
    params = NODETrainParams(input_params_file)
    path_to_input = joinpath(input_params_file, "..")
    path_to_output = joinpath(input_params_file, "..", "..", "output_data", params.train_id)
    params.input_data_path = path_to_input
    params.output_data_path = path_to_output
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
        plots_loss, plots_obs, plots_pred =
            visualize_3(params, path_to_output, path_to_input, visualize_level)
        for (i, p) in enumerate(plots_loss)
            Plots.png(p, joinpath(path_to_output, string("_loss_", i)))
        end
        for (i, p) in enumerate(plots_obs)
            Plots.png(p, joinpath(path_to_output, string("_obs_", i)))
        end
        for (i, p) in enumerate(plots_pred)
            Plots.png(p, joinpath(path_to_output, string("_pred_", i)))
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
    plots_loss = []
    plots_obs = []
    plots_pred = []
    p1 = Plots.plot(df_loss.Loss, title = "Loss")   #TODO, Plot the last 5%, last 10% of loss values. Use log scale... 
    p2 = Plots.plot(df_loss.RangeCount, title = "Range Count")
    last_5_percent = Int(ceil(length(df_loss.Loss) * 0.05))
    last_10_percent = Int(ceil(length(df_loss.Loss) * 0.10))
    p3 = Plots.plot(df_loss.Loss[(end - last_5_percent):end], title = "Last 5% of loss")
    p4 = Plots.plot(df_loss.Loss[(end - last_10_percent):end], title = "Last 10% of loss")
    p = Plots.plot(p1, p2, p3, p4, layout = (2, 2))
    push!(plots_loss, p)

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
        preds = df_predictions[i, "prediction"]
        obs = df_predictions[i, "observation"]
        t_preds = df_predictions[i, "t_prediction"]
        n_total = Int(size(preds[1])[1] / size(t_preds[1])[1])
        n_observable = n_total - params.node_unobserved_states
        obs = [reshape(o, (n_observable, Int(length(o) / n_observable))) for o in obs]
        preds = [reshape(p, (n_total, Int(length(p) / n_total))) for p in preds]
        n_total = size(preds[1])[1]
        ground_truth = concatonate_ground_truth(fault_data, df_loss[i, :PVS_name], :)
        ir_true = ground_truth[1, :]    #TODO, generalize to more ground truth states (not necessarily currents)
        ii_true = ground_truth[2, :]
        t_all = concatonate_t(tsteps, df_loss[i, :PVS_name], :)
        p3 = Plots.scatter(t_all', ir_true, ms = 2, msw = 0, label = "truth")
        p4 = Plots.scatter(t_all', ii_true, ms = 2, msw = 0, label = "truth")

        for (i, t_pred) in enumerate(t_preds)
            Plots.scatter!(p3, t_preds[i], obs[i][1, :], ms = 2, msw = 0, legend = false)
            Plots.scatter!(p4, t_preds[i], obs[i][2, :], ms = 2, msw = 0, legend = false)
        end
        p = Plots.plot(
            p3,
            p4,
            title = string(df_loss[i, :PVS_name], " loss: ", output_dict["final_loss"]),
            layout = (2, 1),
        )
        push!(plots_obs, p)

        list_subplots = []
        for i in 1:size(preds[1])[1]
            p = Plots.plot()
            for (j, tpred) in enumerate(t_preds)
                Plots.scatter!(
                    p,
                    t_preds[j],
                    preds[j][i, :],
                    ms = 2,
                    msw = 0,
                    legend = false,
                    title = string("state # ", i),
                )
            end
            push!(list_subplots, p)
        end
        p = Plots.plot(list_subplots...)

        push!(plots_pred, p)
    end
    return plots_loss, plots_obs, plots_pred
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

function print_train_parameter_overview(train_params_folder)
    Matrix = Any[]
    header = Symbol[]
    files = filter(x -> contains(x, ".json"), readdir(train_params_folder, join = true))
    files = filter(x -> contains(x, "train_"), files)
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
                if fieldname == :training_groups    #Special case for compact printing 
                    push!(
                        Matrix_row,
                        [[
                            getfield(params, fieldname)[1][:tspan],
                            getfield(params, fieldname)[1][:shoot_times],
                            getfield(params, fieldname)[1][:multiple_shoot_continuity_term],
                            getfield(params, fieldname)[1][:batching_sample_factor],
                        ]],
                    )
                elseif fieldname == :node_state_inputs  #Special case for compact printing 
                    push!(Matrix_row, length(getfield(params, fieldname)))
                else
                    push!(Matrix_row, getfield(params, fieldname))
                end

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

    common_params = Matrix[:, common_params_indices]
    common_header = header[common_params_indices]
    changing_params = Matrix[:, changing_params_indices]
    changing_header = header[changing_params_indices]

    print("COMMON PARAMETERS:\n")
    PrettyTables.pretty_table(common_params, header = common_header)
    print("CHANGING PARAMETERS:\n")
    PrettyTables.pretty_table(
        changing_params,
        header = changing_header,
        highlighters = (PrettyTables.Highlighter(
            (data, i, j) -> true,
            PrettyTables.Crayon(bold = true, background = :red),
        )),
    )
end
