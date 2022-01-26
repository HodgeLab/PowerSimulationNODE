path = (joinpath(pwd(), "serialize-test-dir"))
!isdir(path) && mkdir(path)
#TODO - clean up this test, check that all fields in the struct are equal. 
try
    test_params = NODETrainParams()
    PowerSimulationNODE.serialize(test_params, "test_params.jl")
    test_params_2 = NODETrainParams("test_params.jl")
    for field_name in fieldnames(NODETrainParams)
        @test getfield(test_params, field_name) == getfield(test_params_2, field_name)
    end 
finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
