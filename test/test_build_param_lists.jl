@testset "Building list of params" begin
    path = (joinpath(pwd(), "build-param-list-test-dir"))
    !isdir(path) && mkdir(path)

    try
        base_option = TrainParams(rng_seed = 1000)
        grid1 = (:check_validation_loss_iterations, ([2, 4], [4, 5], [5, 6]))
        grid2 = (:initializer_n_layer, (10, 20))
        a = build_grid_search!(base_option, grid1, grid2)
        @test length(a) == 6

        total_runs = 10
        random2 = (:initializer_n_layer, (min = 10, max = 20))
        random3 = (:log_η, (min = -2.0, max = -1.0))
        b = build_random_search!(base_option, total_runs, random2, random3)
        @test length(b) == 10
    finally
        @info("removing test files")
        rm(path, force = true, recursive = true)
    end
end
