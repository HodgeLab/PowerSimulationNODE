path = (joinpath(pwd(), "hpc-test-dir"))
!isdir(path) && mkdir(path)

try
    test = [
        TrainParams(train_id = "test1"),
        TrainParams(train_id = "test2"),
        TrainParams(train_id = "test3"),
    ]

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

    @test occursin("SLURM_ARRAY_TASK_ID", file)

    hpc_params = AlpineHPCTrain(;
        username = "test_user",
        params_data = test,
        project_folder = "test2",
        scratch_path = path,
    )

    mkpath(joinpath(hpc_params.scratch_path, hpc_params.project_folder))

    generate_train_files(hpc_params)
    file = read(hpc_params.train_bash_file, String)
    @test occursin("SLURM_ARRAY_TASK_ID", file)

finally
    @info("removing test files")
    rm(path, force = true, recursive = true)
end
