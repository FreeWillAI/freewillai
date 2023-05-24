import base58
import logging
import binascii
import re
import os
import asyncio
import aioipfs
import asyncio
import aioipfs
import eth_account
from web3 import Web3, HTTPProvider
from typing import Coroutine, Dict, List, Optional, Tuple
from freewillai.constants import FWAI_DIRECTORY 
from freewillai.converter import ModelConverter
from freewillai.doctypes import Abi, Bytecode
from freewillai.exceptions import NotSupportedError, UserRequeriment
from freewillai.globals import Global
from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from solcx import compile_standard
from torch.nn import Module as PytorchModule
from tensorflow.keras import Model as KerasModule


async def add_files(files: list):
    async with aioipfs.AsyncIPFS(host='localhost', port=5001) as client:
        list = []
        async for added_file in client.add(*files, recursive=True):
            logging.debug('Imported file {0}, URL: {1}'.format(
                added_file['Name'], get_url(added_file['Hash'])))
            list.append(added_file)
        return list            


async def cat_file(cid):
    async with aioipfs.AsyncIPFS(host='localhost', port=5001) as client:
        return await client.core.cat(cid)


def save_file(inp,out_path):
    if not os.path.exists(FWAI_DIRECTORY):
        os.mkdir(FWAI_DIRECTORY)
    with open(out_path, 'wb') as file:
        file.write(inp)


def get_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    return loop
    

def task_runner(*tasks: Coroutine) -> List:
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


def get_path(endpoint: str) -> str:
    if 'https://ipfs.io/ipfs/' in endpoint:
        endpoint = get_hash_from_url(endpoint)
    return FWAI_DIRECTORY + endpoint


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


def get_hash_from_bytes32(bytes_array):
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
        raise UserRequeriment(
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


def get_w3():
    return Web3(HTTPProvider(Global.provider_endpoint))


if __name__ == "__main__":
    compile_test_contract('smart_contract/TaskRunner.sol', 'TaskRunner')
