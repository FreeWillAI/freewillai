from typing import Dict, TypedDict, Union, List


Stderr = Union[bytes, str, None]
Stdout = Union[bytes, str, None]

ModelName = str
DatasetName = str

ModelName = str
DatasetName = str


class AbiDict(TypedDict):
    inputs: List[Dict[str, str]]
    name: str
    outputs: List[Dict[str, str]]
    stateMutability: str
    type: str


Abi = List[AbiDict]
Bytecode = str
