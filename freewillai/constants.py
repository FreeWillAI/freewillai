FWAI_DIRECTORY = '/tmp/freewillai_files/'


AVALIABLE_DATASET_FORMATS = [
    'csv',
    'json',
    'zip_images',
    'directory_images',
    'tensor',
    'image',
]


AVALIABLE_MODEL_LIBRARIES = [
    'torch',
    'keras',
    'sklearn',
    'onnx',
]


# Deployed on sepolia
TOKEN_CONTRACT_ADDRESS = '0x5997fB5Cc05Bd53A5fe807eb8BA592d664099d5a'
TOKEN_CONTRACT_ABI_PATH = 'contracts/FreeWillAITokenABI.json'

# Deployed on sepolia
TASK_RUNNER_CONTRACT_ADDRESS = '0x4036E6F21D735128a784Fa3897e8260FAA146ED3'
TASK_RUNNER_CONTRACT_ABI_PATH = 'contracts/TaskRunnerABI.json'
