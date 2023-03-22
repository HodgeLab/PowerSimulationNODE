
const INPUT_FOLDER_NAME = "input_data"
const OUTPUT_FOLDER_NAME = "output_data"
const INPUT_SYSTEM_FOLDER_NAME = "system_data"
const HPC_TRAIN_FILE = "hpc_train.sh"
const HPC_GENERATE_DATA_FILE = "hpc_generate_data.sh"

const PSY_CONSOLE_LEVEL = Logging.Error
const PSY_FILE_LEVEL = Logging.Error

# Make use of these constants for PSID logging if needed 
#const PSID_CONSOLE_LEVEL = Logging.Error
#const PSID_FILE_LEVEL = Logging.Error

const SURROGATE_EXOGENOUS_INPUT_DIM = 2 #[Vr, Vi]
const SURROGATE_SS_INPUT_DIM = 3    #[P, Q, V, θ]  --> changed to [Vq, Id, Iq] (Vq = 0 in local ref frame)
const SURROGATE_OUTPUT_DIM = 2 #[Ir, Ii]
const SURROGATE_N_REFS = 2
const ATTEMPTS_TO_FIND_CONVERGENT_SURROGATE = 10
const NN_INPUT_LIMITS = (min = -1.0, max = 1.0)
const NN_TARGET_LIMITS = (min = -1.0, max = 1.0)

const TIME_LIMIT_BUFFER_SECONDS = 5400 #90 minutes to compile Julia environment (before training) and capture output data (after training) before hpc process is killed externally (increased from 45 mins)

if Sys.iswindows() || Sys.isapple()
    const NODE_CONSOLE_LEVEL = Logging.Info
    const NODE_FILE_LEVEL = Logging.Error
else
    const NODE_CONSOLE_LEVEL = Logging.Error
    const NODE_FILE_LEVEL = Logging.Info
end
