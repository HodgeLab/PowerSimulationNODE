path = (joinpath(pwd(), "serialize-test-dir"))
!isdir(path) && mkdir(path)

try
    test_params = TrainParams()
    PowerSimulationNODE.serialize(test_params, joinpath(path, "test_params.jl"))
    test_params_2 = TrainParams(joinpath(path, "test_params.jl"))
    for field_name in fieldnames(TrainParams)
        @test getfield(test_params, field_name) == getfield(test_params_2, field_name)
    end
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
