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
