from __future__ import annotations

import os
import time
import logging
import numpy as np
import asyncio
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Dict
from numpy import ndarray
from dataclasses import dataclass

from web3 import Web3
from freewillai.common import IPFSBucket, OnnxModel, TestDataset, Task
from freewillai.globals import Global
from freewillai.exceptions import UserRequeriment
from freewillai.utils import get_account, get_modellib, get_path, get_url, get_w3
from freewillai.contract import TaskRunnerContract


@dataclass
class FWAIResult:
    data: ndarray
    url: str


class TaskRunner:

    def __init__(
        self, 
        model, 
        dataset,
        preprocess: Optional[Dict] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        private_key: Optional[str] = None,
        poll_interval: int = 1,
        web3_instance: Optional[Web3] = None,
    ):

        if get_modellib(model) == 'torch' and input_size is None:
            raise UserRequeriment(
                f'{input_size=}. Torch model needs input_size argument.'
            )

        self.model = model
        self.dataset = dataset
        self.preprocess = preprocess
        self.input_size = input_size
        self.poll_interval = poll_interval

        self.model_path = NamedTemporaryFile().name
        self.dataset_path = NamedTemporaryFile().name
        
        self.account = get_account(private_key)
        self.task_runner = TaskRunnerContract(self.account, web3_instance=web3_instance)

        self.model_cid = None
        self.dataset_cid = None
        self.result = None
        self.start_time = 0

        self.mint_for_jupyter_demo()


    ### NOTE: It's just for demo
    def mint_for_jupyter_demo(self):
        from freewillai.contract import testing_mint_to_node
        testing_mint_to_node(self.account.address)


    async def dispatch(self) -> None:
        TestDataset.local_save(self.dataset, self.dataset_path)
        OnnxModel.local_save(self.model, self.preprocess, self.input_size, self.model_path)

        if not (os.path.exists(self.model_path) 
            and os.path.exists(self.dataset_path) 
        ):
            raise RuntimeError(
                'Unexpected error in the existence of the paths. '
                f'{self.model_path=}, {self.dataset_path=}'
            )

        self.model_cid = await IPFSBucket.upload(self.model_path, 'model') 
        self.dataset_cid = await IPFSBucket.upload(self.dataset_path, 'dataset') 


    async def _listen_for_result(self, task):
        timeout = self.task_runner.get_timeout()
        logging.debug(f"\n[*] Pending for result. {timeout=}")
        logging.debug(f"  > Task id: {task.id}")
        logging.debug(f"  > model_url: {task.model_url}")
        logging.debug(f"  > dataset_url: {task.dataset_url}")

        bar = tqdm(range(timeout))
        while self.task_runner.check_within_timewindow(task.id):

            await asyncio.sleep(1)
            bar.update(1)
            actualtime = int(time.time()-task.start_time)
            
            if actualtime > timeout * 1.5:
                raise RuntimeError('There was an error with your task')

            ### TODO: remove it [testing function]
            self.task_runner.generate_block()
        
        return await self.validate(task)


    async def validate(self, task):
        logging.debug('  > Validating results...')
        self.task_runner.validate_task_if_ready(task.id)
        result_url = self.task_runner.get_task_result(task.model_url, task.dataset_url)
        print('  > Result URL:', result_url)

        await IPFSBucket.download(result_url, file_type='result')
        self.result = FWAIResult(
            data=np.load(get_path(result_url)),
            url=result_url
        )
        return self.result


    async def run_and_get_result(self) -> FWAIResult:
        assert self.model_cid and self.dataset_cid
        model_url = get_url(self.model_cid)
        dataset_url = get_url(self.dataset_cid)
        _, event_data = self.task_runner.add_task(model_url, dataset_url)
        task = Task.load_from_event(event_data)
        result = await self._lister_for_result(task)
        return result


async def run_task(
    model,
    dataset, 
    preprocess: Optional[Dict] = None, 
    input_size: Optional[Tuple[int, ...]] = None,
    verbose: bool = False
) -> FWAIResult:

    logging.basicConfig(format="%(levelname)s: %(message)s")
    if verbose:
        logging.basicConfig(
            format="%(message)s", level=logging.DEBUG
        )

    tr = TaskRunner(model, dataset, preprocess, input_size)
    await tr.dispatch()
    return await tr.run_and_get_result()


def connect(provider: str) -> Web3:
    Global.provider_endpoint = provider
    return get_w3()
