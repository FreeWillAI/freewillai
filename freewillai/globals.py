import os
from dataclasses import dataclass
from dotenv import load_dotenv
from freewillai.constants import (
    TASK_RUNNER_CONTRACT_ABI_PATH, TASK_RUNNER_CONTRACT_ADDRESS, 
    TOKEN_CONTRACT_ABI_PATH, TOKEN_CONTRACT_ADDRESS, FWAI_DIRECTORY
)

# Initialization of global variables
# To set variables by default you just need put it in FWAI_DIRECTORY/global.env
def init_env(work_dir=None) -> None:
    if work_dir is None:
        work_dir = FWAI_DIRECTORY
    path = os.path.join(work_dir, "global.env")
    if os.path.exists(path):
        load_dotenv(path)


@dataclass
class Global:
    # Environment Variables 
    working_directory = os.environ.get("FREEWILLAI_WORKING_DIRECTORY") or FWAI_DIRECTORY
    init_env(working_directory)
    provider_endpoint = os.environ.get("FREEWILLAI_PROVIDER") or 'http://127.0.0.1:8545'
    token_address = os.environ.get("FREEWILLAI_TOKEN_ADDRESS") or TOKEN_CONTRACT_ADDRESS
    task_runner_address = os.environ.get("FREEWILLAI_TASK_RUNNER_ADDRESS") or TASK_RUNNER_CONTRACT_ADDRESS
    token_abi_path = os.environ.get("FREEWILLAI_TOKEN_ABI_PATH") or TOKEN_CONTRACT_ABI_PATH
    task_runner_abi_path = os.environ.get("FREEWILLAI_TASK_RUNNER_ABI_PATH") or TASK_RUNNER_CONTRACT_ABI_PATH
    anvil_config_path = os.environ.get("ANVIL_CONFIG_PATH") or None
    ipfs_host = os.environ.get("IPFS_HOST") or "127.0.0.1"
    ipfs_port = os.environ.get("IPFS_PORT") or 5001
    provider = None

    # API Variables
    model_lib = None
    dataset_type = None
    verbose = True

    @classmethod
    def update(cls):
        cls.working_directory = os.environ.get("FREEWILLAI_WORKING_DIRECTORY") or FWAI_DIRECTORY
        init_env(cls.working_directory)
        cls.provider_endpoint = os.environ.get("FREEWILLAI_PROVIDER") or 'http://127.0.0.1:8545'
        cls.token_address = os.environ.get("FREEWILLAI_TOKEN_ADDRESS") or TOKEN_CONTRACT_ADDRESS
        cls.task_runner_address = os.environ.get("FREEWILLAI_TASK_RUNNER_ADDRESS") or TASK_RUNNER_CONTRACT_ADDRESS
        cls.token_abi_path = os.environ.get("FREEWILLAI_TOKEN_ABI_PATH") or TOKEN_CONTRACT_ABI_PATH
        cls.task_runner_abi_path = os.environ.get("FREEWILLAI_TASK_RUNNER_ABI_PATH") or TASK_RUNNER_CONTRACT_ABI_PATH
        cls.anvil_config_path = os.environ.get("ANVIL_CONFIG_PATH") or None
        cls.ipfs_host = os.environ.get("IPFS_HOST") or "127.0.0.1"
        cls.ipfs_port = os.environ.get("IPFS_PORT") or 5001

        # API Variables
        cls.model_lib = None
        cls.dataset_type = None
        cls.verbose = True
