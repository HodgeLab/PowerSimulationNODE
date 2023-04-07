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
#SBATCH --ntasks=1
#
# Memory per cpu
#SBATCH --mem-per-cpu={{mb_per_cpu}}M
#
# Processors per task (for future parallel training code):
#SBATCH --cpus-per-task={{n_cpus_per_task}}
#
#SBATCH --time={{time_limit}}
#SBATCH --output={{{train_path}}}/job_output_%A_%a.o
#SBATCH --error={{{train_path}}}/job_output_%A_%a.e
#SBATCH --array=1-{{n_tasks}}

export TMPDIR={{{train_path}}}/tmp/
# Check Dependencies
julia --project={{{project_path}}} -e 'using Pkg; Pkg.instantiate()'

INFILE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" {{{train_set_file}}})

julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/train_node.jl \$INFILE 
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
#SBATCH --ntasks=1 
#
# Memory per cpu
#SBATCH --mem-per-cpu={{mb_per_cpu}}M
#
# Processors per task (for future parallel training code):
#SBATCH --cpus-per-task={{n_cpus_per_task}}
#
#SBATCH --time={{time_limit}}
#SBATCH --output={{{train_path}}}/job_output_%A_%a.o
#SBATCH --error={{{train_path}}}/job_output_%A_%a.e
#SBATCH --array=1-{{n_tasks}}

export TMPDIR={{{train_path}}}/tmp/
# Check Dependencies
julia --project={{{project_path}}} -e 'using Pkg; Pkg.instantiate()'
julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/build_subsystems.jl {{{first_parameter_path}}}

INFILE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" {{{generate_data_set_file}}})

julia --project={{{project_path}}} {{{project_path}}}/scripts/hpc_train/generate_data.jl \$INFILE 
"""

struct HPCTrain
    username::String
    account::String
    QoS::String
    partition::String
    project_folder::String
    train_folder::String
    scratch_path::String
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
    train_folder_for_data::Union{String, Nothing}
end

"""
    function SavioHPCTrain(;
        username,
        params_data,
        project_folder = "PowerSystemNODEs",
        scratch_path = "/global/scratch/users",
        time_limit_train = "0-24:00:00",
        n_tasks = 1,
        QoS = "savio_normal",
        partition = "savio",
        train_folder_for_data = nothing,
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
    time_limit_train = "0-23:59:59",
    time_limit_generate_data = "00:30:00",
    QoS = "savio_normal",
    partition = "savio",
    train_folder_for_data = nothing,
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
        train_folder_for_data,
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
        time_limit_train = "0-24:00:00",
        n_tasks = 1,  #default to parallelize across all tasks 
        QoS = "normal",
        partition = "amilan",
        train_folder_for_data = nothing,
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
    time_limit_train = "0-23:59:59",
    time_limit_generate_data = "00:30:00",
    QoS = "normal",
    partition = "amilan",
    train_folder_for_data = nothing,
    mb_per_cpu = 9600,
)
    time_format = Dates.DateFormat("d-H:M:S")
    for p in params_data
        t = Dates.Time(time_limit_train, time_format)
        z = Dates.Time("0-00:00:00", time_format)
        p.train_time_limit_seconds = floor((t - z).value * 10^-9)
    end

    # Default until we parallelize training code
    n_cpus_per_task = 1
    return HPCTrain(
        username,
        "ucb340_asc1",
        QoS,
        partition,
        project_folder,
        train_folder,
        scratch_path,
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
        train_folder_for_data,
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
    train_bash_file = train.train_bash_file
    if train.train_folder_for_data === nothing
        generate_data_bash_file = train.generate_data_bash_file
        generate_data_job_id = readchomp(`sbatch --parsable $generate_data_bash_file`)
        return run(`sbatch --dependency=afterok:$generate_data_job_id $train_bash_file`)
    else
        copy_data_and_subsystems(train)
        return run(`sbatch $train_bash_file`)
    end
end

function copy_data_and_subsystems(train::HPCTrain)
    train_folder_for_data = train.train_folder_for_data
    train_folder = train.train_folder

    train_data_paths = unique([p.train_data_path for p in train.params_data])
    validation_data_paths = unique([p.validation_data_path for p in train.params_data])
    test_data_paths = unique([p.test_data_path for p in train.params_data])
    surrogate_system_path = unique([p.surrogate_system_path for p in train.params_data])
    train_system_path = unique([p.train_system_path for p in train.params_data])

    for path in train_data_paths
        name = basename(path)
        @assert name in readdir(joinpath(train_folder_for_data, INPUT_FOLDER_NAME))
        cp(
            joinpath(train_folder_for_data, INPUT_FOLDER_NAME, name),
            joinpath(train_folder, INPUT_FOLDER_NAME, name),
            force = true,
        )
    end
    for path in validation_data_paths
        name = basename(path)
        @assert name in readdir(joinpath(train_folder_for_data, INPUT_FOLDER_NAME))
        cp(
            joinpath(train_folder_for_data, INPUT_FOLDER_NAME, name),
            joinpath(train_folder, INPUT_FOLDER_NAME, name),
            force = true,
        )
    end
    for path in test_data_paths
        name = basename(path)
        @assert name in readdir(joinpath(train_folder_for_data, INPUT_FOLDER_NAME))
        cp(
            joinpath(train_folder_for_data, INPUT_FOLDER_NAME, name),
            joinpath(train_folder, INPUT_FOLDER_NAME, name),
            force = true,
        )
    end

    @assert length(surrogate_system_path) == 1
    @assert length(train_system_path) == 1

    for path in surrogate_system_path
        name = basename(path)
        validation_descriptors_name =
            string(split(name, ".")[1], "_validation_descriptors.", split(name, ".")[2])
        @assert name in readdir(joinpath(train_folder_for_data, INPUT_SYSTEM_FOLDER_NAME))
        cp(
            joinpath(train_folder_for_data, INPUT_SYSTEM_FOLDER_NAME, name),
            joinpath(train_folder, INPUT_SYSTEM_FOLDER_NAME, name),
            force = true,
        )
        cp(
            joinpath(
                train_folder_for_data,
                INPUT_SYSTEM_FOLDER_NAME,
                validation_descriptors_name,
            ),
            joinpath(train_folder, INPUT_SYSTEM_FOLDER_NAME, validation_descriptors_name),
            force = true,
        )
    end
    for path in train_system_path
        name = basename(path)
        validation_descriptors_name =
            string(split(name, ".")[1], "_validation_descriptors.", split(name, ".")[2])
        @assert name in readdir(joinpath(train_folder_for_data, INPUT_SYSTEM_FOLDER_NAME))
        cp(
            joinpath(train_folder_for_data, INPUT_SYSTEM_FOLDER_NAME, name),
            joinpath(train_folder, INPUT_SYSTEM_FOLDER_NAME, name),
            force = true,
        )
        cp(
            joinpath(
                train_folder_for_data,
                INPUT_SYSTEM_FOLDER_NAME,
                validation_descriptors_name,
            ),
            joinpath(train_folder, INPUT_SYSTEM_FOLDER_NAME, validation_descriptors_name),
            force = true,
        )
    end
    cp(
        joinpath(
            train_folder_for_data,
            INPUT_SYSTEM_FOLDER_NAME,
            "connecting_branches_names",
        ),
        joinpath(train_folder, INPUT_SYSTEM_FOLDER_NAME, "connecting_branches_names"),
        force = true,
    )
    return
end
