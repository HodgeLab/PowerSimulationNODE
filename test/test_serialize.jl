path = (joinpath(pwd(), "serialize-test-dir"))
!isdir(path) && mkdir(path)

try
    test_params = TrainParams()
    PowerSimulationNODE.serialize(test_params, joinpath(path, "test_params.jl"))
    test_params_2 = TrainParams(joinpath(path, "test_params.jl"))
    #Spot check
    @test test_params.train_data.operating_points[1] ==
          test_params_2.train_data.operating_points[1]
    display(test_params)
    display(test_params_2)
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
