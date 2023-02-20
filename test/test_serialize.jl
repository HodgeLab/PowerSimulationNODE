@testset "Test serializing and reading back TrainParams" begin
    path = (joinpath(pwd(), "serialize-test-dir"))
    !isdir(path) && mkdir(path)

    try
        test_params = TrainParams()
        PowerSimulationNODE.serialize(test_params, joinpath(path, "test_params.jl"))
        test_params_2 = TrainParams(joinpath(path, "test_params.jl"))
        #Spot check
        @test test_params.train_data.operating_points[1] ==
              test_params_2.train_data.operating_points[1]
        @test test_params.optimizer == test_params_2.optimizer
        for f in fieldnames(typeof(test_params.model_params))
            @test getfield(test_params.model_params, f) ==
                  getfield(test_params_2.model_params, f)
        end

    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end
