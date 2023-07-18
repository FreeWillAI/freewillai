import base58
import logging
import binascii
import re
import os
import asyncio
import aioipfs
import asyncio
import aioipfs
import json
import eth_account
import torch
import numpy as np
import random
from web3 import Web3, HTTPProvider
from typing import Coroutine, Dict, List, Optional, Tuple, Literal
from freewillai.converter import ModelConverter
from freewillai.doctypes import Abi, Bytecode
from freewillai.exceptions import NotSupportedError, UserRequirement
from freewillai.globals import Global
from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from solcx import compile_standard
from torch.nn import Module as PytorchModule
from tensorflow.keras import Model as KerasModule
from tempfile import NamedTemporaryFile
import zipfile
from pathlib import Path

async def add_files(files: list):
    async with aioipfs.AsyncIPFS(host=Global.ipfs_host, port=int(Global.ipfs_port)) as client:
        lst = []
        async for added_file in client.add(*files, recursive=True):
            logging.debug('Imported file {0}, URL: {1}'.format(
                added_file['Name'], get_url(added_file['Hash'])))
            lst.append(added_file)
        return lst            


async def cat_file(cid: str):
    async with aioipfs.AsyncIPFS(host=Global.ipfs_host, port=int(Global.ipfs_port)) as client:
        return await client.core.cat(cid)


def save_file(inp, filename, mode="wb"):
    if not os.path.exists(Global.working_directory):
        os.mkdir(Global.working_directory)
    out_path = os.path.join(Global.working_directory, filename)
    with open(out_path, mode) as file:
        file.write(inp)


def in_cache(cid_or_url: str) -> bool:
    if not os.path.exists(Global.working_directory):
        os.mkdir(Global.working_directory)
    if 'ipfs' in cid_or_url:
        cid_or_url = get_hash_from_url(cid_or_url)
    return cid_or_url in os.listdir(Global.working_directory)


def get_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    return loop
    

def async_runner(*tasks: Coroutine) -> List:
    async def _main():
        global values
        values = await asyncio.gather(*tasks) 

    loop = get_loop()
    loop.run_until_complete(_main())
    return values


def get_hash_from_url(url: str) -> str:
    return re.findall(r'ipfs\/(.*)', url)[0]


def get_url(hsh: str) -> str:
    return f'https://ipfs.io/ipfs/{hsh}'


def get_path(cid_or_url: str) -> str:
    if not os.path.exists(Global.working_directory):
        os.mkdir(Global.working_directory)
    if 'https://ipfs.io/ipfs/' in cid_or_url:
        cid_or_url = get_hash_from_url(cid_or_url)
    return Global.working_directory + cid_or_url


def get_convertion_func(model):
    if isinstance(model, PytorchModule):
        return ModelConverter.pytorch_to_onnx

    elif isinstance(model, KerasModule):
        return ModelConverter.keras_to_onnx

    elif "<class 'sklearn" in repr(type(model)):
        return ModelConverter.sklearn_to_onnx

    elif 'onnx' in repr(type(model)):
        if 'InferenceSession' in repr(type(model)):
            raise NotSupportedError('Onnx model can\'t be an InferenceSession')
        return ModelConverter.dump_onnx

    else:
        raise NotSupportedError('Model library is not supported yet')


def get_modellib(model):
    if isinstance(model, PytorchModule):
        return 'torch'
    elif isinstance(model, KerasModule):
        return 'keras'
    elif "<class 'sklearn" in repr(type(model)):
        return 'sklearn'
    elif 'onnx' in repr(type(model)):
        return 'onnx'
    else:
        return None


def get_bytes32_from_hash(hsh: str):
    bytes_array = base58.b58decode(hsh)
    b = bytes_array[2:]
    hex = binascii.hexlify(b).decode('utf-8')
    return Web3.to_bytes(hexstr=hex)


def get_hash_from_bytes32(bytes_array) -> str:
    merged = 'Qm' + bytes_array
    return base58.b58encode(merged).decode('utf-8')


def get_account(private_key=None) -> LocalAccount:
    load_dotenv()
    if private_key is None:
        private_key = os.environ.get('PRIVATE_KEY')

    try:
        return eth_account.Account.from_key(private_key)

    except ValueError as err:
        raise ValueError(err)

    except:
        raise UserRequirement(
            "PRIVATE_KEY is required as environment variable.\n"
            "Please set your the private key by following one of these ways:\n"
            "  > executing a bash command: export PRIVATE_KEY='paste-your-private-key-here'\n"
            "  > Write into the .env file: PRIVATE_KEY='paste-your-private-key-here'"
        )

def compile_test_contract(
    contract_path: str, contract_name: str, settings: Optional[Dict] = None
) -> Tuple[Abi, Bytecode]:

    with open(contract_path, 'r') as contract_file:
        contract_code = contract_file.read()

    if settings is None:
        settings = {
            "language": "Solidity",
            "sources": {
                contract_path: {
                    "content": contract_code
                }
            },
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode"]
                    }
                }
            }
        }

    os.chdir('smart_contract/')
    compiled_contract = compile_standard(settings) 
    os.chdir('..')

    abi = compiled_contract['contracts'][contract_path][contract_name]['abi']
    bytecode = compiled_contract['contracts'][contract_path][contract_name]['evm']['bytecode']['object']
        
    return abi, bytecode


def get_w3(uri: Optional[str] = None):
    if not uri:
        uri = Global.provider_endpoint
    return Web3(HTTPProvider(uri))


def build_envfile(file_path: str, out_path: str) -> None:

    with open(file_path, "r") as jsonfile:
        data = json.loads(jsonfile.read())

    out_string = ""

    for account, public_key in enumerate(data["available_accounts"]):
        out_string += f"ANVIL_NODE{account + 1}={public_key}\n"

    private_key_builded = False
    for account, private_key in enumerate(data["private_keys"]):
        if not private_key_builded:
            out_string += f"PRIVATE_KEY={private_key}\n"
            private_key_builded = True
            
        out_string += f"ANVIL_NODE{account + 1}_PRIVATE_KEY={private_key}\n"

    out_string += f"ANVIL_BASE_FEE={data['base_fee']}\n"
    out_string += f"ANVIL_GAS_LIMIT={data['gas_limit']}\n"
    out_string += f"ANVIL_GAS_PRICE={data['gas_price']}\n"
    out_string += f"ANVIL_MNEMONIC=\"{data['wallet']['mnemonic']}\"\n"
    out_string += f"ANVIL_CONFIG_PATH={file_path}\n"
    
    with open(out_path, "w") as outfile:
        outfile.write(out_string)


def get_private_key_by_id(id: int, file_path: str, file_type: Literal["env", "config"]="config"):
    assert file_type in ["env", "config"], f"Invalid argument: {file_type=}"

    if file_type == "config":
        out_path = NamedTemporaryFile().name
        build_envfile(file_path, out_path)
        file_path = out_path

    load_dotenv(file_path) 
    return os.environ.get(f"ANVIL_NODE{id}_PRIVATE_KEY")


def load_global_env(file_path: str, override: bool = True) -> bool:
    loaded = load_dotenv(file_path, override=override)
    if not loaded:
        return loaded
    
    Global.update()

    return loaded


def zip_directory(directory_to_zip, zip_file_path):
    directory_to_zip = Path(directory_to_zip)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in directory_to_zip.rglob('*'):
            zip_file.write(file, file.relative_to(directory_to_zip))


def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    set_seed(42)
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=1,
        top_p=1,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def deploy_contracts(out_env="anvil.contracts.env"):
    import subprocess
    script_path = '/'.join(__file__.split("/")[:-2]) + "/scripts/deploy_contracts.sh"
    subprocess.check_call(f"{script_path} config {out_env}", shell=True)
