@warn "TESTFILEWRITE"
using CSV
import JSON
using PowerSimulationNODE
#Look at the size/form of the file. 

#Benchmark writing an arrow file

#Benchmark writing a csv file

function _write_csv_from_df(df, file)
    open(file, "w") do io
        CSV.write(io, df)
    end
end

function _capture_output2(output_dict, output_directory, id)
    output_path = joinpath(output_directory, id)
    mkpath(output_path)
    for (key, value) in output_dict
        if typeof(value) == DataFrames.DataFrame
            df = pop!(output_dict, key)
            PowerSimulationNODE._write_arrow_from_df(df, joinpath(output_path, key))    #Address out-of-memory error with function barrier and GC?
            df = nothing
            #GC.gc()
        end
    end
end

#_capture_output(output_dict, output_directory, id)
@time df = DataFrame(rand(2000, 1000), :auto)
@time df = DataFrame(rand(2000, 1000), :auto)
@time PowerSimulationNODE._write_arrow_from_df(df, "test1")
@time PowerSimulationNODE._write_arrow_from_df(df, "test2")
@time _write_csv_from_df(df, "test1.csv")
@time _write_csv_from_df(df, "test2.csv")

##
output_dict = PowerSimulationNODE._initialize_output_dict()
output_dict["parameters"] = DataFrame(rand(2000, 1000), :auto)
output_dict["predictions"] = DataFrame(rand(2000, 1000), :auto)
output_dict["loss"] = DataFrame(rand(2000, 1000), :auto)

@time PowerSimulationNODE._capture_output(output_dict, "test1", "1")

output_dict = PowerSimulationNODE._initialize_output_dict()
output_dict["parameters"] = DataFrame(rand(2000, 1000), :auto)
output_dict["predictions"] = DataFrame(rand(2000, 1000), :auto)
output_dict["loss"] = DataFrame(rand(2000, 1000), :auto)

@time _capture_output2(output_dict, "test2", "1")
