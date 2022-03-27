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

"""
    function visualize_training(input_params_file::String; visualize_level=1)

Visualize a single training by generating plots. `visualize_level` controls the number and type of plots
- `visualize_level = 1`: plot end result only (the result used to calculate final loss)
- `visualize_level = 2`: plot when moving to new fault(s)
- `visualize_level = 3`: plot when moving to new data range
- `visualize_level = 4`: plot every iteration moving to new data range (Do not use with trainings with many iterations)

# NOTE: this functions assumes that the file resides in `input_data` directory and that there is a corresponding `output_data` directory with the training outputs. 
# Examples:
````
visualize_training("train_1.json", visualize_level = 3)
````
"""
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
        #Cleanup to be able to delete the arrow files: https://github.com/apache/arrow-julia/issues/61
        GC.gc()
        return
    end
end

function read_arrow_file_to_dataframe(file::AbstractString)
    return open(file, "r") do io
        DataFrames.DataFrame(Arrow.Table(io))
    end
end

"""
    function animate_training(input_params_file::String; skip_frames=10, fps = 10)

Visualize a single training by generating animation of the training process. Saves a gif in `output_data` for predictions and observations.
- `skip_frames`: number of training iterations to skip between frames of the animation. 
- `fps`: frames per second in the output gif

# Examples:
````
animate_training("train_1.json", skip_frames=10, fps = 10)
````
"""
function animate_training(input_params_file::String; skip_frames = 10, fps = 10)
    _animate_training(input_params_file, skip_frames = skip_frames, fps = fps)
    GC.gc()
end

function _animate_training(input_params_file::String; skip_frames = 10, fps = 10)
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

    df_loss = read_arrow_file_to_dataframe(joinpath(path_to_output, "loss"))
    plots_obs = []
    plots_pred = []
    PVS_name_recorded_entries = df_loss.PVS_name[output_dict["recorded_iterations"]]
    transition_indices = collect(1:skip_frames:length(PVS_name_recorded_entries))
    df_predictions = read_arrow_file_to_dataframe(joinpath(path_to_output, "predictions"))
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
        ground_truth = concatonate_ground_truth(fault_data, PVS_name_recorded_entries[i], :)
        ir_true = ground_truth[1, :]    #TODO, generalize to more ground truth states (not necessarily currents)
        ii_true = ground_truth[2, :]
        t_all = concatonate_t(tsteps, PVS_name_recorded_entries[i], :)
        p3 = Plots.scatter(t_all', ir_true, ms = 2, msw = 0, label = "truth")
        p4 = Plots.scatter(t_all', ii_true, ms = 2, msw = 0, label = "truth")

        for (i, t_pred) in enumerate(t_preds)
            Plots.scatter!(p3, t_preds[i], obs[i][1, :], ms = 2, msw = 0, legend = false)
            Plots.scatter!(p4, t_preds[i], obs[i][2, :], ms = 2, msw = 0, legend = false)
        end
        p = Plots.plot(
            p3,
            p4,
            title = string(
                PVS_name_recorded_entries[i],
                " loss: ",
                output_dict["final_loss"],
            ),
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
    anim_obs = Plots.Animation()
    for p in plots_obs[1:(end - 1)]
        p = Plots.plot(p)
        Plots.frame(anim_obs)
    end
    anim_preds = Plots.Animation()
    for p in plots_pred[1:(end - 1)]
        p = Plots.plot(p)
        Plots.frame(anim_preds)
    end

    Plots.gif(anim_obs, joinpath(path_to_output, "anim_obs.gif"), fps = fps)
    Plots.gif(anim_preds, joinpath(path_to_output, "anim_preds.gif"), fps = fps)

    #Cleanup to be able to delete the arrow files: https://github.com/apache/arrow-julia/issues/61
    df_loss = nothing
    df_predictions = nothing
    GC.gc()
    return
end

function visualize_2(params, path_to_output, path_to_input)
    df_loss = read_arrow_file_to_dataframe(joinpath(path_to_output, "loss"))
    p1 = Plots.plot(df_loss.Loss, title = "Loss")
    p2 = Plots.plot(df_loss.RangeCount, title = "Range Count")
    return Plots.plot(p1, p2, layout = (2, 1))
end

function visualize_3(params, path_to_output, path_to_input, visualize_level)
    df_loss = read_arrow_file_to_dataframe(joinpath(path_to_output, "loss"))
    plots_loss = []
    plots_obs = []
    plots_pred = []
    p1 = Plots.plot(df_loss.Loss[1:(end - 1)], yaxis = :log, title = "Loss")
    p2 = Plots.plot(df_loss.RangeCount, title = "Range Count")
    last_5_percent = Int(ceil(length(df_loss.Loss) * 0.05))
    last_10_percent = Int(ceil(length(df_loss.Loss) * 0.10))
    p3 = Plots.plot(
        df_loss.Loss[(end - last_5_percent):(end - 1)],
        title = "Last 5% of loss",
    )
    p4 = Plots.plot(
        df_loss.Loss[(end - last_10_percent):(end - 1)],
        title = "Last 10% of loss",
    )
    p = Plots.plot(p1, p2, p3, p4, layout = (2, 2))
    push!(plots_loss, p)

    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    PVS_name_recorded_entries = df_loss.PVS_name[output_dict["recorded_iterations"]]
    RangeCount_recorded_entries = df_loss.RangeCount[output_dict["recorded_iterations"]]

    if visualize_level == 1
        transition_indices = [length(PVS_name_recorded_entries)]
    elseif visualize_level == 2
        transition_indices = find_transition_indices(PVS_name_recorded_entries)
    elseif visualize_level == 3
        transition_indices = find_transition_indices(RangeCount_recorded_entries)
    elseif visualize_level == 4
        transition_indices = collect(1:length(PVS_name_recorded_entries))
    else
        @warn "Invalid value for parameter visualize_level"
    end
    df_predictions = read_arrow_file_to_dataframe(joinpath(path_to_output, "predictions"))
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
        ground_truth = concatonate_ground_truth(fault_data, PVS_name_recorded_entries[i], :)
        ir_true = ground_truth[1, :]    #TODO, generalize to more ground truth states (not necessarily currents)
        ii_true = ground_truth[2, :]
        t_all = concatonate_t(tsteps, PVS_name_recorded_entries[i], :)
        p3 = Plots.scatter(t_all', ir_true, ms = 2, msw = 0, label = "truth")
        p4 = Plots.scatter(t_all', ii_true, ms = 2, msw = 0, label = "truth")

        for (i, t_pred) in enumerate(t_preds)
            Plots.scatter!(p3, t_preds[i], obs[i][1, :], ms = 2, msw = 0, legend = false)
            Plots.scatter!(p4, t_preds[i], obs[i][2, :], ms = 2, msw = 0, legend = false)
        end
        p = Plots.plot(
            p3,
            p4,
            title = string(
                RangeCount_recorded_entries[i],
                " loss: ",
                output_dict["final_loss"],
            ),
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

"""
    function generate_summary(output_data_path)

Generates a plot with high level information about a collection of trainings with outputs in `output_data_path`. Visualizations include:
- plot of total iterations vs total time
- plot of total trainable parameters vs total time
# Examples:
````
d = generate_summary("output_data")
visualize_summary(d)
````
"""
function generate_summary(output_data_path)
    output_directories = readdir(output_data_path)
    high_level_outputs_dict = Dict{String, Dict{String, Any}}()
    for dir in output_directories
        output_dict = JSON3.read(
            read(joinpath(output_data_path, dir, "high_level_outputs")),
            Dict{String, Any},
        )
        high_level_outputs_dict[output_dict["train_id"]] = output_dict
    end
    return high_level_outputs_dict
end

function print_high_level_output_overview(
    high_level_outputs_dict,
    path;
    ignore_keys = ["train_id"],
)
    for (key, value) in high_level_outputs_dict
        for (k, v) in value
            if in(k, ignore_keys)
                pop!(high_level_outputs_dict[key], k)
            end
        end
    end
    open(joinpath(path, "HighLevelOverview.txt"), "w") do io
        print(io, "HIGH LEVEL OVERVIEW:\n")
        PrettyTables.pretty_table(io, high_level_outputs_dict, limit_printing = false)
    end
end

function visualize_summary(high_level_outputs_dict)
    p1 = Plots.scatter()
    p2 = Plots.scatter()
    p = Plots.plot()
    for (key, value) in high_level_outputs_dict
        Plots.scatter!(
            p1,
            (value["total_time"], value["final_loss"]),
            label = value["train_id"],
            xlabel = "total time (s)",
            ylabel = "final loss",
            yaxis = :log,
            markersize = 1,
            markerstrokewidth = 0,
        )
        Plots.annotate!(
            p1,
            value["total_time"],
            value["final_loss"],
            Plots.text(value["train_id"], :red, 3),
        )
        Plots.scatter!(
            p2,
            (value["total_time"], value["n_params"]),
            label = value["train_id"],
            xlabel = "total time (s)",
            ylabel = "n params nn",
            yaxis = :log,
            markersize = 1,
            markerstrokewidth = 0,
        )
        Plots.annotate!(
            p2,
            value["total_time"],
            value["n_params_nn"],
            Plots.text(value["train_id"], :red, 3),
        )
        p = Plots.plot(p1, p2)
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

"""
    function print_train_parameter_overview(input_data_path)

Prints tables with both constant and changing parameters for a collection of train parameter files in `input_data_path`
# Examples:
````
print_train_parameter_overview("input_data")
````
"""
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
            ]
            if !(fieldname in exclude_fields)
                if fieldname == :training_groups    #Special case for compact printing 
                    push!(
                        Matrix_row,
                        [[
                            length(getfield(params, fieldname)),
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
    open(joinpath(train_params_folder, "TrainParameterOverview.txt"), "w") do io
        print(io, "COMMON PARAMETERS:\n")
        PrettyTables.pretty_table(
            io,
            common_params,
            header = common_header,
            limit_printing = false,
        )
        print(io, "CHANGING PARAMETERS:\n")
        PrettyTables.pretty_table(
            io,
            changing_params,
            header = changing_header,
            highlighters = (PrettyTables.Highlighter(
                (data, i, j) -> true,
                PrettyTables.Crayon(bold = true, background = :red),
            )),
            limit_printing = false,
        )
    end
end
