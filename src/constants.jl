
const INPUT_FOLDER_NAME = "input_data"
const OUTPUT_FOLDER_NAME = "output_data"
const INPUT_SYSTEM_FOLDER_NAME = "system_data"
const HPC_TRAIN_FILE = "hpc_train.sh"

const PSY_CONSOLE_LEVEL = Logging.Error
const PSY_FILE_LEVEL = Logging.Info

const PSID_CONSOLE_LEVEL = Logging.Error
const PSID_FILE_LEVEL = Logging.Error

const SURROGATE_EXOGENOUS_INPUT_DIM = 2 #[Vr, Vi]
const SURROGATE_SS_INPUT_DIM = 4    #[P, Q, V, θ]
const SURROGATE_OUTPUT_DIM = 2 #[Ir, Ii]

if Sys.iswindows() || Sys.isapple()
    const NODE_CONSOLE_LEVEL = Logging.Info
    const NODE_FILE_LEVEL = Logging.Error
else
    const NODE_CONSOLE_LEVEL = Logging.Error
    const NODE_FILE_LEVEL = Logging.Info
end
