from __future__ import annotations

import json
import re
import os
import onnx
import logging
import subprocess
import numpy as np
import torch
import shutil
import tensorflow as tf
import polars as pl
from typing import Literal, Dict, Optional
from dataclasses import dataclass

from freewillai.utils import (
    add_files, cat_file, get_convertion_func, save_file, get_hash_from_url, 
    get_path, get_account, get_w3
)
from freewillai.constants import AVALIABLE_DATASET_FORMATS, FWAI_DIRECTORY
from freewillai.exceptions import NotSupportedError
from freewillai.contract import TaskRunnerContract
from PIL import Image as PILImage
from numpy import ndarray
from csv import Sniffer


class IPFSBucket:

    @staticmethod
    async def upload(
        path: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ):
        if file_type:
            logging.debug(f'\n[*] Uploading {file_type}...')
        hsh = await add_files([path])
        hsh = hsh[0]['Hash']
        return hsh

    @staticmethod
    async def download(
        url: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ):
        if file_type:
            logging.debug(f'\n[*] Downloading {file_type}...')

        func = lambda x : x
        if 'ipfs' in url:
            func = get_hash_from_url

        hsh = func(url)
        file = await cat_file(hsh)
        save_file(file, FWAI_DIRECTORY + hsh)


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
    ) -> None:
        if isinstance(dataset, ndarray):
            np.save(output_path, dataset)
            shutil.move(output_path + '.npy', output_path)

        elif isinstance(dataset, (torch.Tensor, tf.Tensor)):
            np.save(output_path, dataset.numpy())
            shutil.move(output_path + '.npy', output_path)

        elif isinstance(dataset, PILImage.Image):
            np.save(output_path, np.array(dataset))
            shutil.move(output_path + '.npy', output_path)

        elif isinstance(dataset, str):
            shutil.copyfile(dataset, output_path)

        else:
            raise NotSupportedError(f'{type(dataset)=}. This dataset type is not supported yet')
        

    def numpy(self) -> ndarray:

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
            'JPEG image': self._from_image,
            'PNG image': self._from_image,
        }
        return get_data_from_format[format]().astype(np.float32)


    def _from_csv(self) -> ndarray:
        with open(self.path, 'r') as csvfile:
            content = csvfile.read()
            delimiter = str(Sniffer().sniff(content).delimiter)
            skip_header = int(Sniffer().has_header(content))
            has_header = Sniffer().has_header(content)

        # return np.genfromtxt(self.path, delimiter=delimiter, skip_header=skip_header)

        lazy_data = pl.scan_csv(
            self.path, 
            separator=delimiter,
            has_header=has_header,
            infer_schema_length=0
        ).with_columns(pl.all().cast(pl.Float32))

        return lazy_data.collect().to_numpy()


    def _from_image(self) -> ndarray:
        image = PILImage.open(self.path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        dataset = np.expand_dims(np.array(image), 0)

        return dataset / 255


    def _from_numpy(self) -> ndarray:
        return np.load(self.path)


class OnnxModel:
    def __init__(self, url: str):
        import onnxruntime as rt
        self.path = get_path(url)
        self.inference_sess = rt.InferenceSession(self.path)
        self.input_name = self.inference_sess.get_inputs()[0].name
        self.label_name = self.inference_sess.get_outputs()[0].name

    def inference(self, dataset: TestDataset):
        assert isinstance(dataset, TestDataset)

        preds = self.inference_sess.run([self.label_name], {self.input_name: dataset.numpy()})
        if isinstance(preds, list):
            preds = preds[0]
        return preds

    @staticmethod
    def local_save(model, preprocess, input_size, model_path):
        get_convertion_func(model)(model, model_path, input_size)
        model = onnx.load(model_path)
        if not preprocess is None:
            for key, value in preprocess.items():
                model.metadata_props.append(
                    onnx.StringStringEntryProto(key=key, value=str(value))
                )
        onnx.save(model, model_path)


@dataclass
class Task:
    id: int
    model_url: str
    dataset_url: str
    start_time: int
    result_url: Optional[str] = None

    def __post_init__(self):
        self.ipfs_url = 'https://ipfs.io/ipfs/'
        self.assert_arguments()

    def assert_arguments(self):
        assert (isinstance(self.id, int)
            and self.id >= 0
        ), f"{self.id=}. {self.id} is not correct task index"

        assert (self.model_url.startswith(self.ipfs_url) 
            and len(self.model_url) == 67
        ), f"{self.model_url=}. {self.model_url} is not correct ipfs url"

        assert (self.dataset_url.startswith(self.ipfs_url) 
            and len(self.dataset_url) == 67
        ), f"{self.dataset_url=}. {self.dataset_url} is not correct ipfs url"

        if self.result_url:
            assert (self.result_url.startswith(self.ipfs_url) 
                and len(self.result_url) == 67
            ), f"{self.result_url=}. {self.result_url} is not correct ipfs url"

    @classmethod
    def load_from_event(cls, event) -> Task:
        w3 = get_w3()
        return cls(
            id = event.args.taskIndex,
            model_url = event.args.model_url,
            dataset_url = event.args.dataset_url,
            start_time = w3.eth.get_block(event.blockNumber).timestamp
        )

    def submit_result(self, result_url, account=None) -> None:
        self.result_url = result_url
        if account is None:
            account = get_account()
        contract = TaskRunnerContract(account)
        contract.submit_result(self)
