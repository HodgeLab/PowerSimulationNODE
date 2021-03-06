const bash_file_template = """
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
#SBATCH --output={{{project_path}}}/job_output_%j.o
#SBATCH --error={{{project_path}}}/job_output_%j.e

export TMPDIR={{{project_path}}}/tmp/
# Check Dependencies
julia --project={{{project_path}}} -e 'using Pkg; Pkg.instantiate()'
{{#force_generate_inputs}} julia --project={{{project_path}}} {{{project_path}}}/scripts/prepare_for_train.jl "true" {{/force_generate_inputs}}

# Load Parallel
module load {{gnu_parallel_name}}

{{#n_nodes}}
# --slf is needed to parallelize across all the cores on multiple nodes
dataecho \$SLURM_JOB_NODELIST |sed s/\\,/\\\\n/g > hostfile
{{/n_nodes}}

{{#n_nodes}}parallel --jobs \$SLURM_CPUS_ON_NODE --slf hostfile \\ {{/n_nodes}}{{^n_nodes}}parallel --jobs \$SLURM_NPROCS \\{{/n_nodes}}
    --wd {{{project_path}}} \\
    -a {{{train_set_file}}}\\
    --joblog {{{project_path}}}/hpc_train.log \\
    srun --export=all --exclusive -n1 -N1 --mem-per-cpu={{mb_per_cpu}}M --cpus-per-task=1 --cpu-bind=cores julia --project={{{project_path}}} {{{project_path}}}/scripts/train_node.jl {}
"""

struct HPCTrain
    username::String
    account::String
    QoS::String
    partition::String
    project_folder::String
    scratch_path::String
    gnu_parallel_name::String
    n_tasks::Int
    n_nodes::Union{Int, Nothing}
    n_cpus_per_task::Int
    mb_per_cpu::Int
    params_data::Vector{TrainParams}
    time_limit::String
    generate_data_time::String
    train_bash_file::String
    force_generate_inputs::Bool
end

"""
    function SavioHPCTrain(;
        username,
        params_data,
        project_folder = "PowerSystemNODEs",
        scratch_path = "/global/scratch/users",
        time_limit = "24:00:00",
        n_tasks = 1,
        QoS = "savio_normal",
        partition = "savio",
        force_generate_inputs = false,
        n_nodes = nothing, # Use with caution in Savio, it can lead to over subscription of nodes
        mb_per_cpu = 4000,
    )
- Function for generating default `HPCTrain` parameters suitable for Summit HPC at CU Boulder.
"""
function SavioHPCTrain(;
    username,
    params_data,
    project_folder = "PowerSystemNODEs",
    scratch_path = "/global/scratch/users",
    time_limit = "23:59:59",
    generate_data_time = "00:30:00",
    n_tasks = 1,
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
        scratch_path,
        "gnu-parallel",
        n_tasks,
        n_nodes,
        n_cpus_per_task,
        mb_per_cpu,
        params_data,
        time_limit,
        generate_data_time,
        joinpath(scratch_path, project_folder, HPC_TRAIN_FILE),
        force_generate_inputs,
    )
end

# Populated with info from: https://curc.readthedocs.io/en/latest/
"""
    function SummitHPCTrain(;
        username,
        params_data,
        project_folder = "PowerSystemNODEs",
        scratch_path = "/scratch/summit/",
        time_limit = "24:00:00",
        n_tasks = 1,  #default to parallelize across all tasks 
        QoS = "normal",
        partition = "shas",
        force_generate_inputs = false,
        mb_per_cpu = 4800,
    )
- Function for generating default `HPCTrain` parameters suitable for Summit HPC at CU Boulder.
"""
function SummitHPCTrain(;
    username,
    params_data,
    project_folder = "PowerSystemNODEs",
    scratch_path = "/scratch/summit/",
    time_limit = "23:59:59",
    generate_data_time = "00:30:00",
    n_tasks = 1,  #default to parallelize across all tasks 
    QoS = "normal",
    partition = "shas",
    force_generate_inputs = false,
    mb_per_cpu = 4800,
)
    time_format = Dates.DateFormat("H:M:S")
    for p in params_data
        time_limit_timeformat = Dates.Time(time_limit, time_format)
        generate_data_timeformat = Dates.Time(generate_data_time, time_format)
        p.train_time_limit_seconds =
            floor((time_limit_timeformat - generate_data_timeformat).value * 10^-9)
    end
    # Default until we parallelize training code
    n_cpus_per_task = 1
    return HPCTrain(
        username,
        "ucb-general", # Get allocation
        QoS,
        partition,
        project_folder,
        scratch_path,
        "gnu_parallel",
        n_tasks,
        nothing, # Default to nothing on Summit since it doesn't dispatch on ssh login
        n_cpus_per_task,
        mb_per_cpu,
        params_data,
        time_limit,
        generate_data_time,
        joinpath(scratch_path, project_folder, HPC_TRAIN_FILE),
        force_generate_inputs,
    )
end

"""
    function generate_train_files(train::HPCTrain)

- Generates the paths and data required to run a training on HPC.
"""
function generate_train_files(train::HPCTrain)
    scratch_path = train.scratch_path
    project_folder = train.project_folder
    mkpath(joinpath(scratch_path, project_folder, PowerSimulationNODE.INPUT_FOLDER_NAME))
    mkpath(
        joinpath(
            scratch_path,
            project_folder,
            PowerSimulationNODE.INPUT_SYSTEM_FOLDER_NAME,
        ),
    )
    mkpath(joinpath(scratch_path, project_folder, PowerSimulationNODE.OUTPUT_FOLDER_NAME))
    touch(joinpath(scratch_path, project_folder, PowerSimulationNODE.HPC_TRAIN_FILE))

    data = Dict()
    data["username"] = train.username
    data["account"] = train.account
    data["QoS"] = train.QoS
    data["time_limit"] = train.time_limit
    data["partition"] = train.partition
    data["gnu_parallel_name"] = train.gnu_parallel_name
    data["project_path"] = joinpath(train.scratch_path, train.project_folder)
    data["n_tasks"] = train.n_tasks
    data["n_cpus_per_task"] = train.n_cpus_per_task
    data["mb_per_cpu"] = train.mb_per_cpu

    data["n_nodes"] = train.n_nodes
    if !isnothing(train.n_nodes)
        data["n_nodes"] = train.n_nodes
    end

    data["force_generate_inputs"] = train.force_generate_inputs
    train_set_folder = joinpath(train.scratch_path, train.project_folder)
    if !ispath(train_set_folder)
        mkpath(train_set_folder)
    end
    data["train_set_file"] = joinpath(train_set_folder, "train_files.lst")

    open(data["train_set_file"], "w") do file
        for param in train.params_data
            param_file_path = joinpath(
                train.scratch_path,
                train.project_folder,
                PowerSimulationNODE.INPUT_FOLDER_NAME,
                "train_$(param.train_id).json",
            )
            if !isfile(param_file_path)
                touch(param_file_path)
            end
            serialize(param, param_file_path)
            write(file, "$param_file_path\n")
        end
    end

    data["force_generate_inputs"] = train.force_generate_inputs
    open(train.train_bash_file, "w") do io
        write(io, Mustache.render(bash_file_template, data))
    end
    return
end

function run_parallel_train(train::HPCTrain)
    bash_file = train.train_bash_file
    return run(`sbatch $bash_file`)
end
