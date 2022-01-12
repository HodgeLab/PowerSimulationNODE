using Mustache
using Logging
using PowerSimulationNODE

struct TestParams
    train_id::String
end

test = [NODETrainParams(train_id = "test2"), NODETrainParams(train_id = "test2")]

hpc_params = SavioHPCTrain(;
    username = "test_user",
    params_data = test,
    project_folder = "test",
    scratch_path = mktempdir(),
    n_nodes = 2,
)
mkpath(joinpath(hpc_params.scratch_path, hpc_params.project_folder))
cd(joinpath(hpc_params.scratch_path, hpc_params.project_folder))

generate_train_files(hpc_params)
file = read(hpc_params.train_bash_file, String)
@test occursin("--slf hostfile", file)
@test !occursin("SLURM_NPROCS", file)

hpc_params = SavioHPCTrain(;
    username = "test_user",
    params_data = test,
    project_folder = "test",
    scratch_path = mktempdir(),
)

generate_train_files(hpc_params)
file = read(hpc_params.train_bash_file, String)
@test !occursin("--slf hostfile", file)
@test occursin("SLURM_NPROCS", file) 
