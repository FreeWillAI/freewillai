from __future__ import annotations

import json
import re
import os
from eth_account.signers.local import LocalAccount
import onnx
import onnxruntime as rt
import numpy as np
import torch
import shutil
import tensorflow as tf
import polars as pl
import subprocess
from typing import Literal, Dict, Optional, Union, List, Type, cast
from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.middleware.buffered_gas_estimate import buffered_gas_estimate_middleware
from web3.middleware import geth_poa_middleware
from web3.middleware.signing import construct_sign_and_send_raw_middleware
from web3.types import Middleware as Web3Middleware, Timestamp
from dataclasses import dataclass
from dotenv import load_dotenv
from freewillai.globals import Global
from freewillai.utils import (
    add_files, build_envfile, cat_file, deploy_contracts, get_convertion_func, get_modellib, load_global_env, 
    save_file, get_hash_from_url, get_path, get_account, get_w3, zip_directory
)
from freewillai.constants import AVALIABLE_DATASET_FORMATS
from freewillai.exceptions import NotSupportedError, UnexpectedError, UserRequirement
from PIL import Image as PILImage
from torchvision.transforms import ToTensor
from tensorflow import convert_to_tensor
from numpy import ndarray
from csv import Sniffer
import tempfile
from typing import List, Optional, Union
from typing_extensions import Literal


class IPFSBucket:

    @staticmethod
    async def upload(
        path: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ) -> str:
        if file_type:
            print(f'\n[*] Uploading {file_type}...')
        if os.path.isdir(path):
        #items = os.listdir(path)
        #dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
        #if len(dirs) >= 2:
            print("creating zip...")
            zip_directory(path, path+'.zip')
            print("path:", path)
            print("zip created!")
            print("uploading...")
            hsh = await add_files([path+'.zip'])

        else:
            hsh = await add_files([path])
        assert len(hsh) == 1, RuntimeError("Unexpected error with ipfs")
        hsh = hsh[0]['Hash']
        print("Uploaded!!")
        return hsh

    @staticmethod
    async def download(
        cid_or_url: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ):
        if file_type:
            print(f'\n[*] Downloading {file_type}...')

        cid = cid_or_url
        if 'ipfs' in cid_or_url:
            cid = get_hash_from_url(cid_or_url)

        file = await cat_file(cid)
        save_file(file, cid)


class Configs:
    def __init__(self, url: str):
        self.path = get_path(url)
        with open(self.path, 'r') as config_file:
            configs = json.loads(config_file.read())

        assert not configs['dataset_format'] or configs['dataset_format'] in AVALIABLE_DATASET_FORMATS
        assert not configs['dataset_delimiter'] or isinstance(configs['dataset_delimiter'], str)
        assert not configs['input_size'] or isinstance(configs['input_size'], list)

        self.dataset_format = configs['dataset_format']
        self.input_size = configs['input_size']
        self.transform = configs['transform']

    @staticmethod
    def local_save(configs: Dict, output_path: str):
        with open(output_path, 'w') as json_file:
            json.dump(configs, json_file)


class TestDataset:
    def __init__(self, url):
        self.path = get_path(url)
    
    @staticmethod
    def local_save(
        dataset, 
        output_path: str, 
        model,
        is_text_data = False
    ) -> None:
        if isinstance(dataset, ndarray):
            np.save(output_path, dataset)
            shutil.move(output_path + '.npy', output_path)

        elif isinstance(dataset, (torch.Tensor, tf.Tensor)):
            np.save(output_path, dataset.numpy())
            shutil.move(output_path + '.npy', output_path)

        elif isinstance(dataset, PILImage.Image):
            image_array = TestDataset.transform_image(dataset, get_modellib(model))
            np.save(output_path, image_array)
            shutil.move(output_path + '.npy', output_path)
        elif is_text_data:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dataset)            

        elif isinstance(dataset, str):
            shutil.copyfile(dataset, output_path)

        else:
            raise NotSupportedError(f'{type(dataset)=}. This dataset type is not supported yet')
        

    @staticmethod
    def transform_image(
        image: Union[PILImage.Image, ndarray],
        modellib
        # input_size: Tuple[int, ...] = None,
    ) -> ndarray:
        
        # Compute transforms by model lib
        get_tensor = {
            "torch": lambda image: ToTensor()(image).numpy(),
            "keras": lambda image: convert_to_tensor(image).numpy(),
            "sklearn": lambda image: np.array(image),
            "onnx": lambda image: np.array(image),
        }
        return get_tensor.get(modellib)(image)

        
    def numpy(self, meta: List[onnx.StringStringEntryProto]) -> ndarray:
    
        configs = {}
        for conf in meta:
            if conf.key == "preprocess" or conf.key == "input_size":
                configs.update({conf.key: json.loads(conf.value)})

            configs.update({conf.key: conf.value})

        if not os.path.exists(self.path):
            raise RuntimeError(f'{self.path=}. Path not found')

        proc = subprocess.Popen(['file', self.path], stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()
        format = ' '.join(re.findall(
            fr'\b\w+\b', stdout.decode('utf-8').replace(f'{self.path}: ', '')
        )[:2]).strip()

        get_data_from_format = {
            'NumPy data': self._from_numpy,
            'CSV text': self._from_csv,
            'ASCII text': self._from_csv,
            'JPEG image': lambda: self._from_image(configs["modellib"]),
            'PNG image': lambda: self._from_image(configs["modellib"]),
        }
        # TODO: Develop a python filetype detector
        f = get_data_from_format.get(format) or self._from_numpy
        return f().astype(np.float32)


    def _from_csv(self) -> ndarray:
        with open(self.path, 'r') as csvfile:
            content = csvfile.read()
            delimiter = str(Sniffer().sniff(content).delimiter)
            skip_header = int(Sniffer().has_header(content))
            has_header = Sniffer().has_header(content)

        # Maybe is better
        # return np.genfromtxt(self.path, delimiter=delimiter, skip_header=skip_header)

        lazy_data = pl.scan_csv(
            self.path, 
            separator=delimiter,
            has_header=has_header,
            infer_schema_length=0
        ).with_columns(pl.all().cast(pl.Float32))

        return lazy_data.collect().to_numpy()


    def _from_image(self, modellib) -> ndarray:
        image = PILImage.open(self.path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_array = self.transform_image(image, modellib)
        dataset = np.expand_dims(image_array, 0)

        return dataset / 255


    def _from_numpy(self) -> ndarray:
        return np.load(self.path)


class OnnxModel:
    def __init__(self, url: str):
        self.path = get_path(url)
        self.inference_sess = rt.InferenceSession(self.path)
        self.input_name = self.inference_sess.get_inputs()[0].name
        self.label_name = self.inference_sess.get_outputs()[0].name

    def inference(self, dataset: TestDataset):
        assert isinstance(dataset, TestDataset)
        
        meta = onnx.load(self.path).metadata_props
        preds = self.inference_sess.run(
            [self.label_name], {self.input_name: dataset.numpy(meta)}
        )
        if isinstance(preds, list):
            preds = preds[0]

        return preds.astype(np.float32)

    @staticmethod
    def local_save(model, preprocess, input_size, model_path, dataset):
        get_convertion_func(model)(model, model_path, input_size)
        onnx_model = onnx.load(model_path)

        if not preprocess is None:
            for key, value in preprocess.items():
                onnx_model.metadata_props.append(
                    onnx.StringStringEntryProto(
                        key="preprocess", value=json.dumps({key: str(value)})
                    )
                )

        onnx_model.metadata_props.extend([
            onnx.StringStringEntryProto(key="modellib", value=get_modellib(model)),
            onnx.StringStringEntryProto(key="input_size", value=json.dumps(input_size)),
        ])
        onnx.save(onnx_model, model_path)


@dataclass
class ContractNodeResult:
    url: str
    sender: str
    node_stake: int

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  url = {self.url}\n"
            f"  sender = {self.sender}\n"
            f"  node_stake = {self.node_stake}\n"
            f")\n"
        )


@dataclass
class FWAIResult:
    data: Union[ndarray, str]
    url: str

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  data = {self.data},\n"
            f"  url = {self.url}\n"
            f")\n"
        )


@dataclass
class Task:
    id: int
    model_url: str
    dataset_url: str
    start_time: Optional[Timestamp] = None
    result_url: Optional[str] = None
    block_number: Optional[int] = None

    def __post_init__(self):
        self.ipfs_url = 'https://ipfs.io/ipfs/'
        self.assert_arguments()

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  id = {self.id}\n"
            f"  model_url = {self.model_url}\n"
            f"  dataset_url = {self.dataset_url}\n"
            f"  result_url = {self.result_url}\n"
            f")\n"
        )


    def assert_arguments(self):
        # Example of url: 
        # https://ipfs.io/ipfs/QmTRzpaMbxQ2pMKS6BXWcZdeQ4aoxKX4ngGn6Qk9WujJo3
        url_valid_lenght = 67

        assert (isinstance(self.id, int)
            and self.id >= 0
        ), f"{self.id=}. {self.id} is not correct task index"

        assert (self.model_url.startswith(self.ipfs_url) 
            and len(self.model_url) == url_valid_lenght
        ), f"{self.model_url=}. {self.model_url} is not correct ipfs url"

        assert (self.dataset_url.startswith(self.ipfs_url)
            and len(self.dataset_url) == url_valid_lenght
        ), f"{self.dataset_url=}. {self.dataset_url} is not correct ipfs url"

        if self.result_url:
            assert (self.result_url.startswith(self.ipfs_url) 
                and len(self.result_url) == url_valid_lenght
            ), f"{self.result_url=}. {self.result_url} is not correct ipfs url"

    @classmethod
    def load_from_event(cls, event, provider: Optional[Provider] = None) -> Task:
        provider = provider or Global.provider or Provider.build()
        w3 = provider.connect()
        return cls(
            id = event['args']['taskIndex'],
            model_url = event['args']['model_url'],
            dataset_url = event['args']['dataset_url'],
            start_time = w3.eth.get_block(event['blockNumber']).get('timestamp'),
            block_number = w3.eth.get_block(event['blockNumber'])
        )

    def submit_result(
        self, 
        result_url: str,
        task_runner_contract: "TaskRunnerContract",
    ) -> None:
        self.result_url = result_url
        tx_hash = task_runner_contract.submit_result(self)
        task_runner_contract.wait_for_transaction(tx_hash)


@dataclass
class AnvilAccount:
    public_key: str
    private_key: str


class Anvil:
    def __init__(self, config_path, uri="http://127.0.0.1:8545", build_envfile=False):
        self.config_path = config_path
        self.uri = uri

        with open(config_path, "r") as jsonfile:
            self.config = json.loads(jsonfile.read())

        if build_envfile:
            self.build_envfile()

        self.accounts = self.config["available_accounts"]
        self.private_keys = self.config["private_keys"]
        self.base_fee = self.config['base_fee']
        self.gas_limit = self.config['gas_limit']
        self.gas_price = self.config['gas_price']
        self.mnemonic = self.config['wallet']['mnemonic']
        self.num_accounts = len(self.accounts)
        self._generate_accounts()

    def _generate_accounts(self):
        for num_node, (account, private_key) in enumerate(zip(self.accounts, self.private_keys)):
            name = f"node{num_node}"
            if num_node == 0:
                name = "master_node"
            self.__setattr__(name, AnvilAccount(account, private_key))

    def build_envfile(self, out_path="/tmp/anvil.env"):
        build_envfile(self.config_path, out_path)
        load_dotenv(out_path)


class Middleware:
    ... 


class Provider:
    url = Global.provider_endpoint
    def __init__(
        self, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
        network_type: Literal["mainnet", "testnet", "non-official"] = "non-official"
    ):
        if network_type == "non-official" and self.uri is None:
            raise UserRequirement(
                "If you want to use a non-official network you need to pass uri argument"
            )

        self.uri = uri
        self.network_type = network_type
        self.token_address = Global.token_address
        self.task_runner_address = Global.task_runner_address

        if (network_type == "non-official" 
            and (
                not self.token_address 
                or not self.task_runner_address
        )):
            raise UserRequirement(
                "If you want to use a non-official network "
                "you need to deploy the contracts on this rpc and export "
                "FREEWILLAI_TOKEN_ADDRESS and FREEWILLAI_TASK_RUNNER_ADDRESS "
                "with these address:"
                "    export FREEWILLAI_TOKEN_ADDRESS=<token-address>"
                "    export FREEWILLAI_TASK_RUNNER_ADDRESS=<task-runner-address>"
            )

        web3_provider_class = HTTPProvider
        if self.uri.startswith("ws"):
            print("[DEBUG] using websocket")
            web3_provider_class = WebsocketProvider
        self.w3 = Web3(web3_provider_class(self.uri), middlewares=middlewares)

        self.add_middleware(
            cast(Web3Middleware, buffered_gas_estimate_middleware),
            'gas_estimate'
        )
        self.network_type = network_type

        if network_type == "testnet":
            print(f"[!] You are connected to a testnet network")
        if network_type == "non-official":
            print(f"[!] You are connected to a non-official network")

    @classmethod
    def name(cls) -> List[str]:
        """
        The name of network
        IMPORTANT: 
            must be in lower case and with the following format:
            testnet or mainnet or devnet / name

        Example:
            'mainnet/scroll'
            'testnet/arbitrum'
            'devnet/anvil'
        """
        return ["default"]

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = []
    ):
        """
        Build provider class with environment variables

        NOTE: Use this try except as good practice
        """
        try:
            cls.asserts()
        except AssertionError as err:
            cls.exception(str(err))

        if uri is None:
            uri = Global.provider_endpoint
        return cls(uri=uri, middlewares=middlewares)

    @classmethod
    def asserts(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = []
    ) -> None:
        """
        Develop the asserts necesaries to ensure well integrate 
        of provider with the rest of code. Including environment variables

        In this case can be FREEWILLAI_PROVIDER 
        or default local provider 'http://localhost:8545'

        Also you can run try except sentences or just asserts.
        If you develop asserts you also need develop an exception method 
        to address this issues or document to user

        IMPORTANT: This code will be run before build() method if build is required


        Arguments
        ---------
            uri: Optional string
                string used to connect provider to network. 
                Also called rpc url for crypto developers
            middlewares: list of middlewares
                Read Middleware class documentation to know more about

            Those arguments are the same as build method
        """
        if (uri is not None
            and (
                not uri.startswith("http://")
                or not uri.startswith("https://")
                or not uri.startswith("ws://")
                or not uri.startswith("wss://"))
        ):
            raise NotSupportedError(
                f"{uri=}. Invalid uri format\n"
                f"uri format: <protocol>://<host>:<port>/<apikey>\n"
                f"  - valid protocols (required): http, https, ws \n"
                f"  - host (required): IP or websites to connect. \n"
                f"    Also you can using localhost\n"
                f"  - port (optional): port to connect\n"
                f"  - apikey (optional): some uri/endpoints needs \n"
                f"    an apikey to able you to interact with them\n"
            )

        if uri is None and not Global.provider_endpoint:
            raise UnexpectedError(
                "For any reason a default uri/rpc-url is not declared. "
                "Please pass a valid uri to fix it"
            )
            
    @classmethod
    def exception(cls, err: str) -> None:
        """
        What to do if assert_env or asserts methods raise errors
        you can: 
            - solve errors occurred by assertions
            - raise UserRequirement error
            - raise Another error with documentation to address it
        """
        raise RuntimeError(err) 

    @classmethod
    def by_network_name(cls, name: str) -> Type[Provider]:
        """Get provider class by name"""
        name = name.lower()
        if name == cls.name():
            return cls
        for subclass in cls.__subclasses__():
            if subclass.name() == name or name in subclass.name():
                return subclass
        raise RuntimeError(
            f"{name=} does not match with any Provider"
        )

    @classmethod
    def is_api_key_required(cls):
        return False

    @property
    def middleware_onion(self) -> Dict:
        return cast(Dict, self.w3.middleware_onion)

    @property
    def middlewares(self) -> List[Web3Middleware]:
        return cast(List[Web3Middleware], self.w3.middleware_onion.middlewares)

    @middlewares.setter
    def middlewares(self, middlewares: Web3Middleware):
        self.w3.middleware_onion.middlewares = middlewares

    def connect(self) -> Web3:
        return self.w3

    def is_mainnet(self):
        return True if self.network_type == "mainnet" else False

    def add_middleware(
        self, 
        middleware: Union[Middleware, Web3Middleware], 
        name: Optional[str] = None, 
        layer: Optional[Literal[0, 1, 2]] = None
    ):
        if layer is None:
            self.w3.middleware_onion.add(middleware, name) 
        else:
            self.w3.middleware_onion.inject(middleware, name, layer)

    def remove_middleware(
        self,
        middleware: Union[Middleware, Web3Middleware, str], 
    ):
        self.w3.middleware_onion.remove(middleware)

    def is_sign_and_send_allowed(self) -> bool:
        if not self.middleware_onion.get("allow_sign_and_send"):
            return False
        return True

    def allow_sign_and_send(self, account: LocalAccount) -> None:
        if not self.is_sign_and_send_allowed():
            self.add_middleware(
                construct_sign_and_send_raw_middleware(account), 'allow_sign_and_send'
            )


class SepoliaProvider(Provider):
    rpc_urls = [
        "https://sepolia.infura.io/v3",
        "https://rpc.sepolia.org",
        "https://rpc2.sepolia.org"
    ]
    env_api_key = ["SEPOLIA_API_KEY", "API_KEY"]
    def __init__(
        self, 
        uri_or_api_key: str,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ):

        if uri_or_api_key.startswith(self.rpc_urls[0]):
            self.api_key = uri_or_api_key.replace(self.url + '/', '')
            self.uri = self.rpc_urls[0] + "/" + self.api_key
        
        elif uri_or_api_key in self.rpc_urls:
            self.uri = uri_or_api_key

        else:
            self.api_key = uri_or_api_key
            self.uri = self.rpc_urls[0] + "/" + self.api_key

        # Contruct uri from api_key
        super().__init__(
            uri=self.uri,
            middlewares=middlewares,
            network_type="testnet",
        )

        self.token_address = "0x5997fB5Cc05Bd53A5fe807eb8BA592d664099d5a"
        self.task_runner_address = "0x4036E6F21D735128a784Fa3897e8260FAA146ED3"

    @classmethod
    def name(cls) -> Union[str, List]:
        return "testnet/sepolia"

    @classmethod
    def _api_key_from_env(cls) -> Optional[str]:
        for env in cls.env_api_key: 
            api_key = os.environ.get(env)
            if not api_key is None:
                return api_key

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ) -> SepoliaProvider:
        try:
            cls.asserts(uri, middlewares)
        except AssertionError as err:
            cls.exception(str(err))

        if (not uri is None 
            and (uri.startswith("http") 
                 or uri.startswith("ws"))
        ):
            api_key = uri.split("/")[-1]
        else:
            api_key = cls._api_key_from_env() or uri

        return cls(cast(str, api_key), middlewares=middlewares)

    @classmethod
    def asserts(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = []
    ) -> None:
        if uri is None and cls._api_key_from_env() is None:
            raise UserRequirement(
                "SEPOLIA_API_KEY is required as environment variable for sepolia.\n"
                "Please set your api key by following one of these ways:\n"
                "  > executing a bash command: export SEPOLIA_API_KEY='paste-your-api-key-here'\n"
                "  > Write into the .env file: SEPOLIA_API_KEY='paste-your-api-key-here'"
            )

        if uri and uri.startswith("https://"):

            if uri.startswith(cls.rpc_urls[0]):
                assert uri.replace(cls.rpc_urls[0], '') != '', f"Invalid {uri=}. Uri must have an API_KEY at the end"
                assert len(uri.split('/')[-1]) >= 32, f"Invalid {uri=}. Invalid api_key"
            else:
                assert uri in cls.rpc_urls, f"Invalid {uri=}. Available rpc_urls: {cls.rpc_urls}"

            assert cls(uri).connect().is_connected(), f"Invalid {uri=}. Cannot connect to web3"

        # This means that uri have to be the api key
        elif uri and not uri.startswith("https://"):
            api_key = uri
            assert len(api_key) >= 32, f"Invalid {api_key=}"
            assert cls(api_key).connect().is_connected(), f"Invalid {api_key=}. Cannot connect to web3"

    @classmethod
    def exception(cls, err: str) -> None:
        raise RuntimeError(err)

    @classmethod
    def is_api_key_required(cls):
        return True


class AnvilProvider(Provider):
    rpc_urls = [
        'http://127.0.0.1:8545'
    ]
    def __init__(
        self, 
        config_path: str,
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ):
        self.uri = uri or self.rpc_urls[0]
        super().__init__(
            uri=self.uri,
            middlewares=middlewares,
            network_type="testnet",
        )
        anvil = Anvil(config_path, uri=self.uri)
        anvil.build_envfile("anvil.env")
        
        anvil_env = "anvil.contracts.env"
        load_global_env(anvil_env)
        lazy_token_address = os.environ.get("FREEWILLAI_TOKEN_ADDRESS")
        if not (
            lazy_token_address is not None
            and self.w3.eth.get_code(Web3.to_checksum_address(lazy_token_address)).hex() != '0x'
        ):
            deploy_contracts(anvil_env)

        self.token_address = os.environ.get("FREEWILLAI_TOKEN_ADDRESS")
        self.task_runner_address = os.environ.get("FREEWILLAI_TASK_RUNNER_ADDRESS")

    @classmethod
    def name(cls) -> Union[str, List]:
        return ["devnet/anvil", "demo"]

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ) -> AnvilProvider:
        config_path = Global.anvil_config_path or "anvil_configs.json"
        uri = uri or cls.rpc_urls[0] 
        return cls(config_path, uri, middlewares=middlewares)


class ScrollTestnetProvider(Provider):
    rpc_urls = [
        # "wss://alpha-rpc.scroll.io/l2"
        "https://alpha-rpc.scroll.io/l2"
    ]
    def __init__(
        self, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ):
        self.uri = uri or self.rpc_urls[0]
        super().__init__(
            uri=self.uri,
            middlewares=middlewares,
            network_type="testnet",
        )
        # For Goerli
        self.add_middleware(
            cast(Web3Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )
        self.token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
        self.task_runner_address = "0x9bDfF39Fa9Bd210629f3FEd4A4470A753268Bb6F"

    @classmethod
    def name(cls) -> Union[str, List]:
        return ["testnet/scroll"]

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ) -> ScrollTestnetProvider:
        uri = uri or cls.rpc_urls[0] 
        return cls(uri, middlewares=middlewares)


class ArbitrumTestnetProvider(Provider):
    rpc_urls = [
        "https://goerli-rollup.arbitrum.io/rpc"
    ]
    def __init__(
        self, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ):
        self.uri = uri or self.rpc_urls[0]
        super().__init__(
            uri=self.uri,
            middlewares=middlewares,
            network_type="testnet",
        )
        # For Goerli
        self.add_middleware(
            cast(Web3Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )
        self.token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
        self.task_runner_address = "0x2aCe1d7797987a23CF35b0695154b5FEBeA42F85"

    @classmethod
    def name(cls) -> Union[str, List]:
        return ["testnet/arbitrum"]

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ) -> ArbitrumTestnetProvider:
        uri = uri or cls.rpc_urls[0] 
        return cls(uri, middlewares=middlewares)
    

class PolygonZkEVMTestnetProvider(Provider):
    rpc_urls = [
        "https://rpc.public.zkevm-test.net"
    ]
    def __init__(
        self, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ):
        self.uri = uri or self.rpc_urls[0]
        super().__init__(
            uri=self.uri,
            middlewares=middlewares,
            network_type="testnet",
        )

        # For Goerli
        self.add_middleware(
            cast(Web3Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )
        
        self.token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
        self.task_runner_address = "0x49717A5D8036be3aaf251463efeCf30bff38A209"

    @classmethod
    def name(cls) -> Union[str, List]:
        return "testnet/polygon/zkevm"

    @classmethod
    def build(
        cls, 
        uri: Optional[str] = None,
        middlewares: List[Union[Web3Middleware, Middleware]] = [],
    ) -> PolygonZkEVMTestnetProvider:
        uri = uri or cls.rpc_urls[0] 
        return cls(uri, middlewares=middlewares)
    

class ETHProvider(Provider):
    @classmethod
    def name(cls) -> Union[str, List]:
        return ["eth", "ethereum"]
