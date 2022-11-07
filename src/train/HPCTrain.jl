const train_bash_file_template = """
#!/bin/bash
# Job name:
#SBATCH --job-name=NODE_train
#
# Account:
#SBATCH --account={{account}}
#
# QoS:
#SBATCH --qos={{QoS}}
#
# Partition:
#SBATCH --partition={{partition}}
#
# Number of MPI tasks requested:
#SBATCH --ntasks={{n_tasks}} \n{{#n_nodes}}#\n#SBATCH --nodes={{n_nodes}} \n {{/n_nodes}}
#
# Memory per cpu
#SBATCH --mem-per-cpu={{mb_per_cpu}}M
#
# Processors per task (for future parallel training code):
#SBATCH --cpus-per-task={{n_cpus_per_task}}
#
#SBATCH --time={{time_limit}}
#SBATCH --output={{{train_path}}}/job_output_%j.o
#SBATCH --error={{{train_path}}}/job_output_%j.e

export TMPDIR={{{train_path}}}/tmp/
# Check Dependencies
julia --project={{{project_path}}} -e 'using Pkg; Pkg.instantiate()'

# Load Parallel
module load {{gnu_parallel_name}}

{{#n_nodes}}
# --slf is needed to parallelize across all the cores on multiple nodes
dataecho \$SLURM_JOB_NODELIST |sed s/\\,/\\\\n/g > hostfile
{{/n_nodes}}

{{#n_nodes}}parallel --jobs \$SLURM_CPUS_ON_NODE --slf hostfile \\ {{/n_nodes}}{{^n_nodes}}parallel --jobs \$SLURM_NPROCS \\{{/n_nodes}}
    --wd {{{train_path}}} \\
    -a {{{train_set_file}}}\\
    --joblog {{{train_path}}}/hpc_train.log \\
    srun --export=all --exclusive -n1 -N1 --mem-per-cpu={{mb_per_cpu}}M --cpus-per-task=1 --cpu-bind=cores julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/train_node.jl {}
"""

const generate_data_bash_file_template = """
#!/bin/bash
# Job name:
#SBATCH --job-name=NODE_generate_data
#
# Account:
#SBATCH --account={{account}}
#
# QoS:
#SBATCH --qos={{QoS}}
#
# Partition:
#SBATCH --partition={{partition}}
#
# Number of MPI tasks requested:
#SBATCH --ntasks={{n_tasks}} \n{{#n_nodes}}#\n#SBATCH --nodes={{n_nodes}} \n {{/n_nodes}}
#
# Memory per cpu
#SBATCH --mem-per-cpu={{mb_per_cpu}}M
#
# Processors per task (for future parallel training code):
#SBATCH --cpus-per-task={{n_cpus_per_task}}
#
#SBATCH --time={{time_limit}}
#SBATCH --output={{{train_path}}}/job_output_%j.o
#SBATCH --error={{{train_path}}}/job_output_%j.e

export TMPDIR={{{train_path}}}/tmp/
# Check Dependencies
julia --project={{{project_path}}} -e 'using Pkg; Pkg.instantiate()'
julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/build_subsystems.jl {{{first_parameter_path}}}

# Load Parallel
module load {{gnu_parallel_name}}

{{#n_nodes}}
# --slf is needed to parallelize across all the cores on multiple nodes
dataecho \$SLURM_JOB_NODELIST |sed s/\\,/\\\\n/g > hostfile
{{/n_nodes}}

{{#n_nodes}}parallel --jobs \$SLURM_CPUS_ON_NODE --slf hostfile \\ {{/n_nodes}}{{^n_nodes}}parallel --jobs \$SLURM_NPROCS \\{{/n_nodes}}
    --wd {{{train_path}}} \\
    -a {{{generate_data_set_file}}}\\
    --joblog {{{train_path}}}/hpc_generate_data.log \\
    srun --export=all --exclusive -n1 -N1 --mem-per-cpu={{mb_per_cpu}}M --cpus-per-task=1 --cpu-bind=cores julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/generate_data.jl {}
"""

struct HPCTrain
    username::String
    account::String
    QoS::String
    partition::String
    project_folder::String
    train_folder::String
    scratch_path::String
    gnu_parallel_name::String
    n_tasks_train::Int
    n_tasks_generate_data::Int
    n_nodes::Union{Int, Nothing}
    n_cpus_per_task::Int
    mb_per_cpu::Int
    params_data::Vector{TrainParams}
    time_limit_train::String
    time_limit_generate_data::String
    generate_data_bash_file::String
    train_bash_file::String
    force_generate_inputs::Bool
end

"""
    function SavioHPCTrain(;
        username,
        params_data,
        project_folder = "PowerSystemNODEs",
        scratch_path = "/global/scratch/users",
        time_limit_train = "24:00:00",
        n_tasks = 1,
        QoS = "savio_normal",
        partition = "savio",
        force_generate_inputs = false,
        n_nodes = nothing, # Use with caution in Savio, it can lead to over subscription of nodes
        mb_per_cpu = 4000,
    )
- Function for generating default `HPCTrain` parameters suitable for Savio HPC at Berkeley.
"""
function SavioHPCTrain(;
    username,
    params_data,
    project_folder = "PowerSystemNODEs",
    train_folder = "train1",
    scratch_path = "/global/scratch/users",
    time_limit_train = "23:59:59",
    time_limit_generate_data = "00:30:00",
    QoS = "savio_normal",
    partition = "savio",
    force_generate_inputs = false,
    n_nodes = nothing, # Use with caution in Savio, it can lead to over subscription of nodes
    mb_per_cpu = 4000,
)

    # Default until we parallelize training code
    n_cpus_per_task = 1
    return HPCTrain(
        username,
        "fc_emac",
        QoS,
        partition,
        project_folder,
        train_folder,
        scratch_path,
        "gnu-parallel",
        1,   #updated during file generation
        1,     #updated during file generation
        n_nodes,
        n_cpus_per_task,
        mb_per_cpu,
        params_data,
        time_limit_train,
        time_limit_generate_data,
        joinpath(scratch_path, project_folder, train_folder, HPC_GENERATE_DATA_FILE),
        joinpath(scratch_path, project_folder, train_folder, HPC_TRAIN_FILE),
        force_generate_inputs,
    )
end

# Populated with info from: https://curc.readthedocs.io/en/latest/
"""
    function AlpineHPCTrain(;
        username,
        params_data,
        project_folder = "PowerSystemNODEs",
        train_folder = "train1",
        scratch_path = "/scratch/alpine/",
        time_limit_train = "24:00:00",
        n_tasks = 1,  #default to parallelize across all tasks 
        QoS = "normal",
        partition = "amilan",
        force_generate_inputs = false,
        mb_per_cpu = 4800,
    )
- Function for generating default `HPCTrain` parameters suitable for Alpine HPC at CU Boulder.
"""
function AlpineHPCTrain(;
    username,
    params_data,
    project_folder = "PowerSystemNODEs",
    train_folder = "train1",
    scratch_path = "/scratch/alpine/",
    time_limit_train = "23:59:59",
    time_limit_generate_data = "00:30:00",
    QoS = "normal",
    partition = "amilan",
    force_generate_inputs = false,
    mb_per_cpu = 9600,
)
    time_format = Dates.DateFormat("H:M:S")
    for p in params_data
        t = Dates.Time(time_limit_train, time_format)
        z = Dates.Time("00:00:00", time_format)
        p.train_time_limit_seconds = floor((t - z).value * 10^-9)
    end

    # Default until we parallelize training code
    n_cpus_per_task = 1
    return HPCTrain(
        username,
        "ucb-general", # Get allocation
        QoS,
        partition,
        project_folder,
        train_folder,
        scratch_path,
        "gnu_parallel",
        1,  #updated during file generation
        1,  #updated during file generation
        nothing, # Default to nothing on Alpine since it doesn't dispatch on ssh login(?)
        n_cpus_per_task,
        mb_per_cpu,
        params_data,
        time_limit_train,
        time_limit_generate_data,
        joinpath(scratch_path, project_folder, train_folder, HPC_GENERATE_DATA_FILE),
        joinpath(scratch_path, project_folder, train_folder, HPC_TRAIN_FILE),
        force_generate_inputs,
    )
end

"""
    function generate_train_files(train::HPCTrain)

- Generates the paths and data required to run a training on HPC.
"""
function generate_train_files(train::HPCTrain)
    path_to_train_folder =
        joinpath(train.scratch_path, train.project_folder, train.train_folder)
    path_to_project_folder = joinpath(train.scratch_path, train.project_folder)
    mkpath(joinpath(path_to_train_folder, "tmp"))
    mkpath(joinpath(path_to_train_folder, PowerSimulationNODE.INPUT_FOLDER_NAME))
    mkpath(joinpath(path_to_train_folder, PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME))
    mkpath(joinpath(path_to_train_folder, PowerSimulationNODE.OUTPUT_FOLDER_NAME))
    touch(joinpath(path_to_train_folder, PowerSimulationNODE.HPC_TRAIN_FILE))

    data_train_template = Dict()
    data_generate_template = Dict()

    data_train_template["username"] = train.username
    data_train_template["account"] = train.account
    data_train_template["QoS"] = train.QoS
    data_train_template["time_limit"] = train.time_limit_train
    data_train_template["partition"] = train.partition
    data_train_template["gnu_parallel_name"] = train.gnu_parallel_name
    data_train_template["project_path"] = path_to_project_folder
    data_train_template["train_path"] = path_to_train_folder
    data_train_template["n_cpus_per_task"] = train.n_cpus_per_task
    data_train_template["mb_per_cpu"] = train.mb_per_cpu
    data_train_template["n_nodes"] = train.n_nodes
    if !isnothing(train.n_nodes)
        data_train_template["n_nodes"] = train.n_nodes
    end

    data_train_template["train_set_file"] =
        joinpath(path_to_train_folder, "train_files.lst")
    open(data_train_template["train_set_file"], "w") do file
        for (i, param) in enumerate(train.params_data)
            param_file_path = joinpath(
                path_to_train_folder,
                PowerSimulationNODE.INPUT_FOLDER_NAME,
                "train_$(param.train_id).json",
            )
            if !isfile(param_file_path)
                touch(param_file_path)
            end
            if i == 1
                data_generate_template["first_parameter_path"] = param_file_path
            end
            serialize(param, param_file_path)
            write(file, "$param_file_path\n")
        end
    end
    data_train_template["n_tasks"] = length(train.params_data)
    open(train.train_bash_file, "w") do io
        write(io, Mustache.render(train_bash_file_template, data_train_template))
    end

    data_generate_template["username"] = train.username
    data_generate_template["account"] = train.account
    data_generate_template["QoS"] = train.QoS
    data_generate_template["time_limit"] = train.time_limit_generate_data
    data_generate_template["partition"] = train.partition
    data_generate_template["gnu_parallel_name"] = train.gnu_parallel_name
    data_generate_template["project_path"] = path_to_project_folder
    data_generate_template["train_path"] = path_to_train_folder
    data_generate_template["n_cpus_per_task"] = train.n_cpus_per_task
    data_generate_template["mb_per_cpu"] = train.mb_per_cpu
    data_generate_template["n_nodes"] = train.n_nodes
    if !isnothing(train.n_nodes)
        data_generate_template["n_nodes"] = train.n_nodes
    end

    data_generate_template["generate_data_set_file"] =
        joinpath(path_to_train_folder, "generate_data_files.lst")
    open(data_generate_template["generate_data_set_file"], "w") do file
        #WRITE UNIQUE TRAIN DATA SETS TO FILE
        train_data_ids = [p.train_data.id for p in train.params_data]
        unique_train_params_data =
            [train.params_data[p] for p in indexin(unique(train_data_ids), train_data_ids)]
        for param in unique_train_params_data
            param_file_path = joinpath(
                path_to_train_folder,
                PowerSimulationNODE.INPUT_FOLDER_NAME,
                "train_$(param.train_id).json",
            )
            if !isfile(param_file_path)
                touch(param_file_path)
            end
            write(file, "$param_file_path,train\n")
        end

        #WRITE UNIQUE VALIDATION DATA SETS TO FILE
        validation_data_ids = [p.validation_data.id for p in train.params_data]
        unique_validation_params_data = [
            train.params_data[p] for
            p in indexin(unique(validation_data_ids), validation_data_ids)
        ]
        for param in unique_validation_params_data
            param_file_path = joinpath(
                path_to_train_folder,
                PowerSimulationNODE.INPUT_FOLDER_NAME,
                "train_$(param.train_id).json",
            )
            if !isfile(param_file_path)
                touch(param_file_path)
            end
            write(file, "$param_file_path,validation\n")
        end

        #WRITE UNIQUE TEST DATA SETS TO FILE
        test_data_ids = [p.test_data.id for p in train.params_data]
        unique_test_params_data =
            [train.params_data[p] for p in indexin(unique(test_data_ids), test_data_ids)]
        for param in unique_test_params_data
            param_file_path = joinpath(
                path_to_train_folder,
                PowerSimulationNODE.INPUT_FOLDER_NAME,
                "train_$(param.train_id).json",
            )
            if !isfile(param_file_path)
                touch(param_file_path)
            end
            write(file, "$param_file_path,test\n")
        end

        #NUMBER OF TASKS IN THE GENERATE BASH FILE IS THE TOTAL NUMBER OF DATASETS TO BE GENERATED 
        data_generate_template["n_tasks"] =
            length(unique_train_params_data) +
            length(unique_validation_params_data) +
            length(unique_test_params_data)
    end

    open(train.generate_data_bash_file, "w") do io
        write(io, Mustache.render(generate_data_bash_file_template, data_generate_template))
    end
    return
end

function run_parallel_train(train::HPCTrain)
    generate_data_bash_file = train.generate_data_bash_file
    train_bash_file = train.train_bash_file
    generate_data_job_id = readchomp(`sbatch --parsable $generate_data_bash_file`)
    return run(`sbatch --dependency=afterok:$generate_data_job_id $train_bash_file`)
end
