path = (joinpath(pwd(), "serialize-test-dir"))
!isdir(path) && mkdir(path)

try
    test_params = NODETrainParams()
    PowerSimulationNODE.serialize(test_params, "test_params.jl")
    test_params_2 = NODETrainParams("test_params.jl")
    @test test_params.train_id == test_params_2.train_id
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
