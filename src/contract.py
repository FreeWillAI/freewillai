import logging
from typing import Any, Dict, Tuple, Optional
from eth_account.signers.local import LocalAccount
from eth_typing import Hash32
from web3 import Web3
from web3.datastructures import AttributeDict
from web3.middleware import construct_sign_and_send_raw_middleware
from freewillai.doctypes import Abi, Bytecode
from freewillai.exceptions import UserRequeriment
from freewillai.utils import get_account, get_w3
from freewillai.constants import (
    TASK_RUNNER_CONTRACT_ABI_PATH, TASK_RUNNER_CONTRACT_ADDRESS, 
    TOKEN_CONTRACT_ADDRESS,TOKEN_CONTRACT_ABI_PATH
)


class Contract:
    def __init__(
        self, 
        account: LocalAccount,
        address: Optional[str] = None, 
        abi_path: Optional[str] = None, 
        abi: Optional[Abi] = None,
        bytecode: Optional[Bytecode] = None,
        constructor_args: Tuple[Any] = tuple(),
        web3_instance: Optional[Web3] = None,
    ):
        assert abi_path or abi
        assert bytecode or address
        
        self.abi = abi
        self.bytecode = bytecode
        self.address = address
        self.w3 = web3_instance

        if self.abi is None and not abi_path is None:
            with open(abi_path) as abifile:
                self.abi = abifile.read()

        if self.w3 is None:
            self.w3 = get_w3()

        assert self.w3.is_connected(), "Not connected to Ethereum node"

        if not bytecode is None:
            self.contract = self.w3.eth.contract(bytecode=self.bytecode, abi=self.abi)
            tx_hash = self.contract.constructor(*constructor_args).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.address = tx_receipt['contractAddress']

        if not self.address is None:
            self.address = Web3.to_checksum_address(self.address)
            self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

        self.account = account
        self.w3.middleware_onion.add(construct_sign_and_send_raw_middleware(self.account))

    @property
    def eth_balance(self):
        return self.w3.eth.get_balance(self.account.address)

    def _get_params(self, gas=3_000_000) -> Dict:
        params = {
            'from': self.account.address,
            'gas': gas,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        }
        return params


class TokenContract(Contract):
    def __init__(
        self, 
        account, 
        bytecode: Optional[Bytecode] = None, 
        abi: Optional[Abi] = None,
        web3_instance: Optional[Web3] = None,
    ):
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=TOKEN_CONTRACT_ADDRESS, 
            abi_path=TOKEN_CONTRACT_ABI_PATH, 
            web3_instance=web3_instance
        )

    @property
    def fwai_balance(self):
        return self.get_balance_of(self.account.address)

    def approve(self, address, amount) -> None:
        tx_params = self._get_params()
        self.contract.functions.approve(address, amount).transact(tx_params)

    def get_balance_of(self, address) -> float:
        return self.contract.functions.balanceOf(address).call()

    def initialize(self) -> None:
        tx_params = self._get_params()
        self.contract.functions.initialize().transact(tx_params)

    def mint(self, address, amount) -> None:
        tx_params = self._get_params()
        self.contract.functions.mint(address, amount).transact(tx_params)


class TaskRunnerContract(Contract):
    def __init__(
        self, 
        account, 
        token_contract_address=TOKEN_CONTRACT_ADDRESS,
        bytecode: Optional[Bytecode] = None, 
        abi: Optional[Abi] = None,
        web3_instance: Optional[Web3] = None,
    ):
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=TASK_RUNNER_CONTRACT_ADDRESS, 
            abi_path=TASK_RUNNER_CONTRACT_ABI_PATH, 
            constructor_args=(token_contract_address,),
            web3_instance=web3_instance
        )
        self.token = TokenContract(account)
        self.task_price = self.contract.functions.taskPrice().call()

    def check_within_timewindow(self, task_id: int):
        return self.send_view_function('checkWithinTimeWindow', task_id)

    def get_staking_amount(self, address):
        return self.contract.functions.stakingAmounts(address).call()

    def send_transact_function(self, func_name: str, *args, **kwargs) -> Hash32:
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args, **kwargs)
        tx_params = self._get_params()
        tx_hash = contract_function.transact(tx_params)
        return tx_hash

    def send_view_function(self, func_name: str, *args, **kwargs): 
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args, **kwargs)
        return contract_function.call()

    def get_timeout(self):
        return self.send_view_function('taskTimeWindow')


    def add_task(self, model_url: str, dataset_url: str) -> Tuple[Hash32, AttributeDict]:
        self.token.approve(self.address, self.task_price)

        tx_hash = self.send_transact_function('addTask', model_url, dataset_url)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        start_time = self.w3.eth.get_block(receipt['blockNumber']).timestamp

        logging.debug(f'{tx_hash=}, {self.task_price=}')
        
        try:
            event_data = self.contract.events.TaskAdded().process_receipt(receipt)[0]
        except IndexError:
            raise UserRequeriment(
                f'Not enough FWAI tokens to add task. Please buy {self.task_price} as minimum'
            )

        logging.debug(f"\n[*] Task added")
        logging.debug(f"  > Transaction hash: {Web3.to_hex(tx_hash)}")
        return tx_hash, event_data

    def get_avaialble_task_results_count(self, task_id) -> int:
        return self.send_view_function('getAvaialbleTaskResultsCount', task_id)

    def validate_all_tasks_if_ready(self) -> Hash32:
        return self.send_transact_function('validateAllTasksIfReady')

    def validate_task_if_ready(self, task_id: int) -> Hash32:
        return self.send_transact_function('validateTaskIfReady', task_id)

    def get_tasks_count(self) -> int:
        available_tasks_count = self.send_view_function('getAvaialbleTasksCount')
        logging.debug(f'[*] Available tasks count = {available_tasks_count}')
        return available_tasks_count

    def get_task_result(self, model_url, dataset_url) -> str:
        return self.send_view_function('getTaskResult', model_url, dataset_url)

    def get_task_time_left(self, task_id) -> int:
        return self.send_view_function('getTaskTimeLeft', task_id)

    def get_returns(self, transaction_hash):
        logging.debug('[*] Waiting transaction mining')
        return self.w3.eth.wait_for_transaction_receipt(
            transaction_hash
        )['logs'][0]['data']
    
    def submit_result(self, task):

        tx_hash = self.send_transact_function(
            'submitResult', 
            task.id, task.model_url, task.dataset_url, task.result_url
        )
        print('id >>>', task.id)
        print('model >>>', task.model_url)
        print('data >>>', task.dataset_url)
        print('result >>>', task.result_url)

        print(f"\n[*] Result Submitted")
        print(f"  > Transaction hash: {Web3.to_hex(tx_hash)}")
        return tx_hash

    def generate_block(self):
        """Testing func"""
        self.send_transact_function('validateAllTasksIfReady')

    def stake(self, amount):
        tx_params = self._get_params()
        self.contract.functions.stake(amount).transact(tx_params)

    def unstake(self, amount):
        tx_params = self._get_params()
        self.contract.functions.unstake(amount).transact(tx_params)


### TODO: !!! Remove it
def testing_mint_to_node(node_address):
    """
    Testing:
        Mint 1000 fwai to any nodes
    """
    owner_private_key = '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80'
    owner_account = get_account(owner_private_key)
    token = TokenContract(owner_account)
    token.initialize()
    fwai_to_mint = 1000 - token.get_balance_of(node_address)
    token.mint(node_address, fwai_to_mint)
