
function apply_to_columns(function_for_column, tsteps)#, y)
    output = Array{Float64}(undef, size(function_for_column(tsteps[1]))[1], length(tsteps))
    for i in 1:length(tsteps)
        output[:, i] = function_for_column(tsteps[i])
    end
    return output
end

function plot_overview(surrogate_prediction, fault_index, fault_data, exs)
    if haskey(surrogate_prediction, :r0)
        return _plot_overview_data(surrogate_prediction, fault_index, fault_data, exs)
    else
        return _plot_overview_physical(surrogate_prediction, fault_index, fault_data, exs)
    end
end

function _plot_overview_physical(surrogate_prediction, fault_index, fault_data, exs)
    if surrogate_prediction.converged
        tsteps = fault_data[fault_index[end][1]].tsteps
        ground_truth_real_current = fault_data[fault_index[end][1]].real_current
        ground_truth_imag_current = fault_data[fault_index[end][1]].imag_current
        t_series = surrogate_prediction.t_series
        i_series = surrogate_prediction.i_series
        i_series = reshape(i_series, (2, length(t_series)))       #Loses shape when serialized/deserialized to arrow  
        res = surrogate_prediction.res
        ex = exs[fault_index[end][1]]

        p1 = Plots.plot(t_series, i_series[1, :], label = L"$\hat{y}_1$")
        Plots.scatter!(p1, t_series, i_series[1, :], label = false, markersize = 1)
        Plots.scatter!(
            [0.0],
            [i_series[1, 1]],
            color = :black,
            label = false,
            markersize = 1,
        )
        Plots.plot!(tsteps, ground_truth_real_current[1, :], label = L"$y_1$")

        p2 = Plots.plot(t_series, i_series[2, :], label = L"$\hat{y}_2$")
        Plots.scatter!(p2, t_series, i_series[2, :], label = false, markersize = 1)
        Plots.scatter!(
            [0.0],
            [i_series[2, 1]],
            color = :black,
            label = false,
            markersize = 1,
        )
        Plots.plot!(tsteps, ground_truth_imag_current[1, :], label = L"$y_2$")
        V = apply_to_columns(ex, t_series)#, i_series)
        V_0 = apply_to_columns(ex, t_series)#, zero(i_series))
        p3 = Plots.plot(t_series, V[1, :], label = L"$u_1$")
        Plots.plot!(t_series, V_0[1, :], label = L"$u_1^0$")
        p4 = Plots.plot(t_series, V[2, :], label = L"$u_2$")
        Plots.plot!(t_series, V_0[2, :], label = L"$u_2^0$")
        return Plots.plot(
            p3,
            p4,
            p1,
            p2,
            layout = (2, 2),
            dpi = 300,
            title = string("f:", fault_index[end][1], ", t:", fault_index[end][2]),
        )
    else
        return Plots.plot()
    end
end

function _plot_overview_data(surrogate_prediction, fault_index, fault_data, exs)
    if size(surrogate_prediction.r0) != size(surrogate_prediction.r_series)
        tsteps = fault_data[fault_index[end][1]].tsteps
        ground_truth_real_current = fault_data[fault_index[end][1]].real_current
        ground_truth_imag_current = fault_data[fault_index[end][1]].imag_current
        lay = Plots.@layout [a{0.3w} [b c; d e]]
        r0_pred = surrogate_prediction.r0_pred
        t_series = surrogate_prediction.t_series
        i_series = surrogate_prediction.i_series
        i_series = reshape(i_series, (2, length(t_series)))       #Loses shape when serialized/deserialized to arrow  
        r_series = surrogate_prediction.r_series
        dim_r = Int(length(r_series) / length(t_series))
        r_series = reshape(r_series, (dim_r, length(t_series)))    #Loses shape when serialized/deserialized to arrow
        res = surrogate_prediction.res
        ex = exs[fault_index[end][1]]

        p1 = Plots.plot(t_series, i_series[1, :], label = L"$\hat{y}_1$")
        Plots.scatter!(p1, t_series, i_series[1, :], label = false, markersize = 1)
        Plots.scatter!(
            [0.0],
            [i_series[1, 1]],
            color = :black,
            label = false,
            markersize = 1,
        )
        Plots.plot!(tsteps, ground_truth_real_current[1, :], label = L"$y_1$")

        p2 = Plots.plot(t_series, i_series[2, :], label = L"$\hat{y}_2$")
        Plots.scatter!(p2, t_series, i_series[2, :], label = false, markersize = 1)
        Plots.scatter!(
            [0.0],
            [i_series[2, 1]],
            color = :black,
            label = false,
            markersize = 1,
        )
        Plots.plot!(tsteps, ground_truth_imag_current[1, :], label = L"$y_2$")
        V = apply_to_columns(ex, t_series)#, i_series)
        V_0 = apply_to_columns(ex, t_series)#, zero(i_series))
        p3 = Plots.plot(t_series, V[1, :], label = L"$u_1$")
        Plots.plot!(t_series, V_0[1, :], label = L"$u_1^0$")
        p4 = Plots.plot(t_series, V[2, :], label = L"$u_2$")
        Plots.plot!(t_series, V_0[2, :], label = L"$u_2^0$")

        p5 = Plots.plot()
        for i in 1:size(r_series, 1)
            Plots.plot!(t_series, r_series[i, :], label = L"r_%$i")
            Plots.scatter!(
                [0.0],
                [r0_pred[i]],
                color = :black,
                label = false,
                markersize = 2,
            )
        end
        return Plots.plot(
            p5,
            p3,
            p4,
            p1,
            p2,
            layout = lay,
            dpi = 300,
            title = string("f:", fault_index[end][1], ", t:", fault_index[end][2]),
        )
    else
        p1 = Plots.scatter(surrogate_prediction.r0_pred, label = "predicted initial values")
        Plots.scatter!(p1, surrogate_prediction.r0, label = "ending initial values")
        p2 = Plots.scatter(
            surrogate_prediction.res,
            label = "residual of SS/Boundary conditions",
            title = string("f:", fault_index[end][1], ", t:", fault_index[end][2]),
        )
        return Plots.plot(p1, p2, layout = (2, 1))
    end
end

"""
    function visualize_training(input_params_file::String, iterations_to_visualize)

Visualize a single training by generating plots. Every `skip` recorded training points will be plotted. 
# NOTE: this functions assumes that the file resides in `input_data` directory and that there is a corresponding `output_data` directory with the training outputs. 
# Examples:
````
visualize_training("train_1.json")
````
"""
function visualize_training(input_params_file::String, iterations_to_visualize)
    params = TrainParams(input_params_file)
    path_to_output = joinpath(input_params_file, "..", "..", "output_data", params.train_id)
    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    println("--------------------------------")
    println("TRAIN ID: ", params.train_id)
    println("TOTAL TIME: ", output_dict["total_time"])
    println("TOTAL ITERATIONS: ", output_dict["total_iterations"])
    println("FINAL LOSS: ", output_dict["final_loss"])
    println("--------------------------------")
    p = _visualize_loss(path_to_output)
    Plots.png(p, joinpath(path_to_output, "loss"))

    p = _visualize_validation_loss(path_to_output)
    Plots.png(p, joinpath(path_to_output, "validation_loss"))

    plots_pred = _visualize_predictions(params, path_to_output, iterations_to_visualize)
    for (i, p) in enumerate(plots_pred)
        Plots.png(p, joinpath(path_to_output, string("_pred_", i)))
    end

    validation_sys = node_load_system(params.modified_surrogate_system_path)
    validation_sys_aux = node_load_system(params.surrogate_system_path)
    validation_dataset = Serialization.deserialize(params.validation_data_path)
    data_collection_location_validation =
        Serialization.deserialize(params.data_collection_location_path)[2]
    df_predictions = read_arrow_file_to_dataframe(joinpath(path_to_output, "predictions"))
    θ = df_predictions[end, "parameters"][1]
    plots_validation_performance = visualize_loss(
        validation_sys,
        validation_sys_aux,
        θ,
        validation_dataset,
        params.validation_data,
        data_collection_location_validation,
        params.model_params,
    )
    for (i, p) in enumerate(plots_validation_performance)
        Plots.png(p, joinpath(path_to_output, string("validation_dataset_", i)))
    end
    #Cleanup to be able to delete the arrow files: https://github.com/apache/arrow-julia/issues/61
    GC.gc()
    return
end

function _visualize_loss(path_to_output)
    df_loss = read_arrow_file_to_dataframe(joinpath(path_to_output, "loss"))
    if minimum(df_loss.Loss_initialization) == 0.0
        p1 = Plots.plot(yaxis = :log)
    else
        p1 = Plots.plot(
            df_loss.Loss_initialization,
            label = "Loss Init",
            yaxis = :log,
            title = "Loss",
        )
    end
    if minimum(df_loss.Loss_dynamic) !== 0.0
        Plots.plot!(p1, df_loss.Loss_dynamic, label = "Loss Dynamic")
    end
    Plots.plot!(p1, df_loss.Loss, label = "Total Loss")
    p2 =
        Plots.plot(df_loss.reached_ss, title = "Steady state/boundary condition converged?")
    return Plots.plot(p1, p2, layout = (2, 1))
end

function _visualize_validation_loss(path_to_output)
    df_validation_loss =
        read_arrow_file_to_dataframe(joinpath(path_to_output, "validation_loss"))
    if DataFrames.nrow(df_validation_loss) == 0
        return Plots.plot(title = "no validation data recorded")
    else
        n_faults = length(df_validation_loss[1, :mae_ir])
        n_evaluations_validation_loss = length(df_validation_loss[!, :mae_ir])
        ir_mean_over_faults = [Statistics.mean(x) for x in df_validation_loss[!, :mae_ir]]
        p1 = Plots.plot(ir_mean_over_faults, label = "mae for ir (across all faults)")
        ir_max_over_faults = [maximum(x) for x in df_validation_loss[!, :max_error_ir]]
        p2 = Plots.plot(ir_max_over_faults, label = "max error for ir (across all faults)")
        ii_mean_over_faults = [Statistics.mean(x) for x in df_validation_loss[!, :mae_ii]]
        p3 = Plots.plot(ii_mean_over_faults, label = "mae for ii (across all faults)")
        ii_max_over_faults = [maximum(x) for x in df_validation_loss[!, :max_error_ii]]
        p4 = Plots.plot(ii_max_over_faults, label = "max error for ii (across all faults)")

        return Plots.plot(
            p1,
            p2,
            p3,
            p4,
            layout = (2, 2),
            title = "number of faults: $(n_faults)",
        )
    end
end

function _visualize_predictions(params, path_to_output, iterations_to_visualize)
    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    recorded_iterations = output_dict["recorded_iterations"]
    df_predictions = read_arrow_file_to_dataframe(joinpath(path_to_output, "predictions"))
    train_dataset = Serialization.deserialize(params.train_data_path)
    exs = _build_exogenous_input_functions(params.train_data, train_dataset)
    plots_pred = []
    for (j, i) in enumerate(recorded_iterations)
        if j in iterations_to_visualize
            surrogate_prediction = df_predictions[j, "surrogate_solution"]
            fault_index = df_predictions[j, "fault_index"]
            p = plot_overview(surrogate_prediction, fault_index, train_dataset, exs)
            push!(plots_pred, p)
        end
    end
    return plots_pred
end

function rebase_path!(input_params_file, new_base_path)
    params = TrainParams(input_params_file)
    params.base_path = new_base_path
    params.system_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        splitpath(params.system_path)[end],
    )
    params.surrogate_system_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        splitpath(params.surrogate_system_path)[end],
    )
    params.modified_surrogate_system_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        splitpath(params.modified_surrogate_system_path)[end],
    )
    params.train_system_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        splitpath(params.train_system_path)[end],
    )
    params.data_collection_location_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        splitpath(params.data_collection_location_path)[end],
    )
    params.train_data_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        splitpath(params.train_data_path)[end],
    )
    params.validation_data_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        splitpath(params.validation_data_path)[end],
    )
    params.test_data_path = joinpath(
        new_base_path,
        PowerSimulationNODE.INPUT_FOLDER_NAME,
        splitpath(params.test_data_path)[end],
    )
    params.output_data_path =
        joinpath(new_base_path, splitpath(params.output_data_path)[end])
    serialize(params, input_params_file)
end

function read_arrow_file_to_dataframe(file::AbstractString)
    return open(file, "r") do io
        DataFrames.DataFrame(Arrow.Table(io))
    end
end

"""
    function animate_training(input_params_file::String; skip=10, fps = 10)

Visualize a single training by generating animation of the training process. Saves a gif in `output_data` for predictions and observations.
- `iterations_to_visualize`:  index of recorded iterations to visualize in the animation.
- `fps`: frames per second in the output gif

# Examples:
````
animate_training("train_1.json", skip=10, fps = 10)
````
"""
function animate_training(input_params_file::String, iterations_to_visualize; fps = 10)
    _animate_training(input_params_file, iterations_to_visualize; fps = fps)
    GC.gc()
end

function _animate_training(input_params_file::String, iterations_to_visualize; fps = 10)
    params = TrainParams(input_params_file)
    path_to_output = joinpath(input_params_file, "..", "..", "output_data", params.train_id)
    output_dict =
        JSON3.read(read(joinpath(path_to_output, "high_level_outputs")), Dict{String, Any})
    println("--------------------------------")
    println("TRAIN ID: ", params.train_id)
    println("TOTAL TIME: ", output_dict["total_time"])
    println("TOTAL ITERATIONS: ", output_dict["total_iterations"])
    println("FINAL LOSS: ", output_dict["final_loss"])
    println("--------------------------------")
    plots_pred = _visualize_predictions(params, path_to_output, iterations_to_visualize)
    anim_preds = Plots.Animation()
    for p in plots_pred
        p = Plots.plot(p)
        Plots.frame(anim_preds)
    end
    Plots.gif(anim_preds, joinpath(path_to_output, "anim_preds.gif"), fps = fps)
    #Cleanup to be able to delete the arrow files: https://github.com/apache/arrow-julia/issues/61
    df_loss = nothing
    df_predictions = nothing
    GC.gc()
    return
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
    function generate_summart(high_level_outputs_dict)

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

"""
    function print_high_level_output_overview(
        high_level_outputs_dict,
        path;
        ignore_keys = ["train_id"],
    )

Print a text file with a high level overview of multiple trainings. 
"""
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

"""
    function generate_summary(output_data_path)
    function generate_summart(high_level_outputs_dict)

Generates a plot with high level information about a collection of trainings with outputs in `output_data_path`. Visualizations include:
- plot of total iterations vs total time
- plot of total trainable parameters vs total time
# Examples:
````
d = generate_summary("output_data")
visualize_summary(d)
````
"""
function visualize_summary(high_level_outputs_dict)
    p1 = Plots.scatter()
    p2 = Plots.scatter()
    p = Plots.plot()
    for (key, value) in high_level_outputs_dict
        if haskey(value["final_loss"], "mae_ir")
            ir_mean = Statistics.mean(value["final_loss"]["mae_ir"])
            ii_mean = Statistics.mean(value["final_loss"]["mae_ii"])
            l = (ir_mean + ii_mean) / 2
            Plots.scatter!(
                p1,
                (value["total_time"], l),
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
                l,
                Plots.text(value["train_id"], :red, 3),
            )
            Plots.scatter!(
                p2,
                (value["total_time"], value["n_params_surrogate"]),
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
                value["n_params_surrogate"],
                Plots.text(value["train_id"], :red, 3),
            )
            p = Plots.plot(p1, p2)
        end
    end
    return p
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
        params = TrainParams(f)
        for fieldname in fieldnames(TrainParams)
            exclude_fields =
                [:base_path, :input_data_path, :output_data_path, :verify_psid_node_off]
            if !(fieldname in exclude_fields)
                if fieldname in [:train_data, :validation_data, :test_data]    #Special case for compact printing 
                    push!(Matrix_row, getfield(params, fieldname)[:id])
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
