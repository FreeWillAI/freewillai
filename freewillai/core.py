from __future__ import annotations

import os
import time
import logging
import numpy as np
import asyncio
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Dict, List, Type, Union

from web3.types import Middleware as Web3Middleware
from eth_account.signers.local import LocalAccount
from freewillai.common import (
    ContractNodeResult, IPFSBucket, OnnxModel, 
    TestDataset, Task, FWAIResult, Provider, Middleware
)
from freewillai.globals import Global
from freewillai.exceptions import FreeWillAIException, UserRequirement
from freewillai.utils import get_account, get_modellib, get_path, get_url, load_global_env
from freewillai.contract import TaskRunnerContract, TokenContract


class TaskRunner:
    def __init__(
        self, 
        model, 
        dataset,
        min_time: int = 1,
        max_time: int = 200,
        min_results: int = 2,
        tokenizer = None,
        preprocess: Optional[Dict] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        private_key_or_account: Optional[Union[str, LocalAccount]] = None,
        poll_interval: int = 1,
        provider: Optional[Provider] = None,
        force_validation: bool = False
    ):
        self.is_text_data = False
        if tokenizer:
            print("HuggingFace LLM")
            self.tokenizer = tokenizer
            self.is_text_data = True

        elif get_modellib(model) == 'torch' and input_size is None:
            raise UserRequirement(
                f'{input_size=}. Torch model needs input_size argument.'
            )

        self.model = model
        self.dataset = dataset
        self.min_time = min_time
        self.max_time = max_time
        self.min_results = min_results
        self.preprocess = preprocess
        self.input_size = input_size
        self.poll_interval = poll_interval
        self.tokenizer = tokenizer
        self.force_validation = force_validation

        self.model_path = NamedTemporaryFile().name
        self.dataset_path = NamedTemporaryFile().name
        
        if isinstance(private_key_or_account, LocalAccount):
            self.account = private_key_or_account
        else:
            self.account = get_account(private_key_or_account)
        
        self.provider = provider or Global.provider or Provider.build()

        self.provider.allow_sign_and_send(self.account)
            
        self.token = TokenContract(
            self.account, 
            address=self.provider.token_address,
            provider=self.provider
        )
        self.task_runner = TaskRunnerContract(
            self.account, 
            address=self.provider.task_runner_address,
            provider=self.provider,
            token_contract=self.token
        )

        self.model_cid = None
        self.dataset_cid = None
        self.result = None
        self.start_time = 0

        ### NOTE: It's just for demo
        ### TODO: Remove it
        if os.environ.get('DEMO'):
            self.prepare_jupyter_demo()

    ### NOTE: It's just for demo
    ### TODO: Remove it
    def prepare_jupyter_demo(self):
        from freewillai.contract import testing_mint_to_node
        testing_mint_to_node(self.account.address, amount=10)

        print(f"\n"
            f'[*] Account balance:\n'
            f'  > {self.token.fwai_balance} FWAI\n'
            f'  > {self.token.eth_balance} ETH\n'
        )


    async def dispatch(self) -> None:
         
        TestDataset.local_save(self.dataset, self.dataset_path, self.model, is_text_data = self.is_text_data) 
        if self.tokenizer:
            print("tokenizer, not onnx")
            os.mkdir(self.model_path)
            print("model_path:",self.model_path,"os exists", (os.path.exists(self.model_path)))
            self.model.save_pretrained(self.model_path + "/model")
            self.tokenizer.save_pretrained(self.model_path + "/tokenizer")
        else:
            OnnxModel.local_save(
                self.model, self.preprocess, self.input_size, 
                self.model_path, self.dataset,
            )
        print("data_set_path", self.dataset_path, "data set exists", os.path.exists(self.dataset_path)) 
        if not (os.path.exists(self.model_path) 
            and os.path.exists(self.dataset_path) 
        ):
            raise RuntimeError(
                'Unexpected error in the existence of the paths. '
                f'{self.model_path=}, {self.dataset_path=}'
            )

        self.model_cid = await IPFSBucket.upload(self.model_path, 'model') 
        self.dataset_cid = await IPFSBucket.upload(self.dataset_path, 'dataset') 


    async def listen_and_validate_response(self, task) -> FWAIResult:
        if Global.verbose:
            print(f"\n"
                f"[*] Waiting for result. "
                f"minTime={self.min_time}s "
                f"maxTime={self.max_time}s "
                f"minResults={self.min_results}"
            )

        bar = tqdm(
            total=self.max_time, 
            bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}| {remaining} left'
        )

        sleep_seconds = .4
        while not self.task_runner.check_if_ready_to_validate(task.id, log=False):

        # Alternative
        # while not self.task_runner.is_in_timeout(task.id, log=False):
        #     if self.task_runner.check_if_ready_to_validate(task.id, log=False):
        #         break

            await asyncio.sleep(sleep_seconds)
            bar.update(sleep_seconds)
            bar_str = str(bar).strip()

            ### NOTE: It's just for demo
            ### TODO: Remove it
            if os.environ.get('DEMO') or True:
                results: List[ContractNodeResult] = (
                    self.task_runner.get_available_task_results(task.id, log=False)
                )
                stdout_str = f"<|TQDM|>:"

                if Global.verbose:
                    # save cpu usage with map
                    results_map = {}
                    for result in results:
                        if not result.url in results_map.keys():
                            await IPFSBucket.download(result.url, file_type='result')

                            ## TODO: choice between csv or numpy to upload a result 
                            # result_array = np.load(get_path(result.url))
                            if self.is_text_data:
                                result_array = open(get_path(result.url), 'r').read()
                            else:
                                result_array = np.loadtxt(get_path(result.url), delimiter=';')

                            results_map[result.url] = result_array
                        stdout_str += (
                            f"[*] Received validation from a node:\n"
                            f"  > result: {result.url}\n"
                             #"  > value: {}\n"
                            f"  > sender: {result.sender}\n"
                        )#.format(results_map.get(result.url) or "pending")

                stdout_str += f"\n{bar_str}"
                print(stdout_str)

            ### TODO: remove it [testing function]
            if 'demo' in self.provider.name():
                self.task_runner.generate_block()
        
        return await self.validate(task)


    async def validate(self, task) -> FWAIResult:
        print('  > Validating results...')

        tx_hash = self.task_runner.validate_task_if_ready(task.id)
        self.task_runner.wait_for_transaction(tx_hash)

        result_url = self.task_runner.get_task_result(task.model_url, task.dataset_url)
        
        if result_url == "":
            import sys
            sys.tracebacklimit = 0
            raise FreeWillAIException(
                    "No consensus reached: not enough nodes agreed on a correct AI result. Please try again with a higher maxTime or lower minResults to allow the network to reach consensus. "
                "You can also check with Free Will AI support about the network status"
            ) from None
        return await self.get_result(result_url)


    async def get_result(self, result_url):
        await IPFSBucket.download(result_url, file_type='result')

        if self.is_text_data:
            with open(get_path(result_url), 'r') as result_file:
                validated_result = result_file.read()
        else:
            validated_result = np.loadtxt(
                get_path(result_url), delimiter=';'
            )

        self.result = FWAIResult(
            data=validated_result,
            url=result_url
        )
        return self.result


    async def run_and_get_result(self) -> FWAIResult:
        assert self.model_cid and self.dataset_cid
        model_url = get_url(self.model_cid)
        dataset_url = get_url(self.dataset_cid)

        if not self.force_validation:
            # Get result if it's already validated
            maybe_result_url = self.task_runner.get_task_result(model_url, dataset_url)
            if maybe_result_url:
                return await self.get_result(maybe_result_url)

        _, event_data = self.task_runner.add_task(
            model_url=model_url,
            dataset_url=dataset_url,
            min_time=self.min_time,
            max_time=self.max_time,
            min_results=self.min_results
        )

        task = Task.load_from_event(event_data, self.provider)
        result = await self.listen_and_validate_response(task)
        return result


async def run_task(
    model,
    dataset, 
    min_time: int = 1,
    max_time: int = 200,
    min_results: int = 2,
    tokenizer = None,
    preprocess: Optional[Dict] = None, 
    input_size: Optional[Tuple[int, ...]] = None,
    verbose: bool = False,
    private_key_or_account: Optional[Union[str, LocalAccount]] = None,
    provider: Optional[Provider] = None,
    force_validation: bool = False
) -> FWAIResult:
    if verbose:
        Global.verbose = True

    if os.environ.get('DEMO'):
        force_validation = True

    tr = TaskRunner(
        model, 
        dataset, 
        min_time,
        max_time,
        min_results,
        tokenizer,
        preprocess, 
        input_size, 
        provider=provider,
        private_key_or_account=private_key_or_account,
        force_validation=force_validation,
    )
    await tr.dispatch()
    return await tr.run_and_get_result()


def connect(
    uri_or_network: str, 
    middlewares: List[Union[Web3Middleware, Middleware]] = [],
    api_key: Optional[str] = None,
    env_file: Optional[str] = None,
    token_address: Optional[str] = None,
    task_runner_address: Optional[str] = None,
) -> Union[Provider, Type[Provider]]:
    env_file = env_file or ".env"
    load_global_env(env_file)

    if uri_or_network.startswith("http") or uri_or_network.startswith("ws"):
        provider = Provider(
            uri_or_network,
            middlewares,
            token_address=token_address,
            task_runner_address=task_runner_address,
        )
    else:
        provider = Provider.by_network_name(uri_or_network)
        try:
            provider.asserts(api_key)
        except AssertionError as err:
            if provider.is_api_key_required() and not api_key:
                raise UserRequirement(
                    f"{err}\n"
                    "Also you can pass api_key as argument, e.g: "
                    "freewillai.connect(\"sepolia\", api_key='paste-your-api-key-here')"
                )
            provider.exception(str(err))

        if provider.is_api_key_required() and api_key:
            os.environ['API_KEY'] = api_key

        provider = provider.build()

    Global.provider = provider
    return provider
