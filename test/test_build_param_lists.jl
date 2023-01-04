@testset "Building list of params" begin
    path = (joinpath(pwd(), "build-param-list-test-dir"))
    !isdir(path) && mkdir(path)

    try
        params_data = TrainParams[]
        no_change_params = Dict{Symbol, Any}()
        change_params = Dict{Symbol, Any}()

        #INDICATE CONSTANT, NON-DEFAULT PARAMETERS
        no_change_params[:rng_seed] = 1

        #INDICATE PARAMETES TO ITERATE OVER COMBINATORIALLY 
        change_params[:validation_loss_every_n] = [1, 2, 3]
        change_params[:train_id] = ["1", "2", "3"]

        build_params_list!(params_data, no_change_params, change_params)
        @test length(params_data) == 9

    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end
