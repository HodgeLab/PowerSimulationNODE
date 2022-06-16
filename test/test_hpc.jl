path = (joinpath(pwd(), "hpc-test-dir"))
!isdir(path) && mkdir(path)

try
    test = [TrainParams(train_id = "test1"), TrainParams(train_id = "test2"), TrainParams(train_id = "test3", data_generation = (operating_points = [(2.0,2.0)], perturbations = "test2"))]

    hpc_params = SavioHPCTrain(;
        username = "test_user",
        params_data = test,
        project_folder = "test",
        scratch_path = path,
        n_nodes = 2,
    )
    mkpath(joinpath(hpc_params.scratch_path, hpc_params.project_folder))

    generate_train_files(hpc_params)
    file = read(hpc_params.train_bash_file, String)

    @test occursin("--slf hostfile", file)
    @test !occursin("SLURM_NPROCS", file)

    hpc_params = SummitHPCTrain(;
        username = "test_user",
        params_data = test,
        project_folder = "test2",
        scratch_path = path,
    )

    mkpath(joinpath(hpc_params.scratch_path, hpc_params.project_folder))

    generate_train_files(hpc_params)
    file = read(hpc_params.train_bash_file, String)
    @test !occursin("--slf hostfile", file)
    @test occursin("SLURM_NPROCS", file)

finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
