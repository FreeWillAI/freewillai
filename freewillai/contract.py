import logging
from typing import Any, Dict, Tuple, Optional, List, cast
from eth_account.signers.local import LocalAccount
from hexbytes import HexBytes
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.datastructures import AttributeDict
from web3.middleware import construct_sign_and_send_raw_middleware
from web3.types import TxParams, TxReceipt
from freewillai.doctypes import Abi, Bytecode
from freewillai.exceptions import UserRequirement, ContractError, Fatal
from freewillai.globals import Global
from freewillai.utils import get_account, get_private_key_by_id
from freewillai.common import Anvil, ContractNodeResult, Provider


class Contract:
    def __init__(
        self, 
        account: LocalAccount,
        address: Optional[str] = None, 
        abi_path: Optional[str] = None, 
        abi: Optional[Abi] = None,
        bytecode: Optional[Bytecode] = None,
        constructor_args: Tuple[Any] = tuple(),
        provider: Optional[Provider] = None,
        allow_sign_and_send: bool = False,
        log = True
    ):
        assert abi_path or abi
        assert bytecode or address
        
        self.abi = abi
        self.bytecode = bytecode
        self.address = address

        # If provider is None build by environment
        self.provider = provider or Global.provider or Provider.build()

        if allow_sign_and_send:
            self.provider.allow_sign_and_send(account)

        if (not allow_sign_and_send
            and self.provider.middleware_onion.get('allow_sign_and_send')
        ):
            import sys
            print(f"[WARNING] Automatic sign and send is allowed in '{self.name}'", file=sys.stderr)

        self.account = account
        self.w3 = self.provider.connect()

        if self.address and self.w3.eth.get_code(self.address).hex() == '0x':
            raise Fatal(
                f"{self.name} not found with this address={self.address}"
            )

        if self.abi is None and not abi_path is None:
            with open(abi_path) as abifile:
                self.abi = abifile.read()

        assert self.w3.is_connected(), "Not connected to Ethereum node"

        if not bytecode is None:
            self.contract = self.w3.eth.contract(bytecode=self.bytecode, abi=self.abi)
            tx_hash = self.contract.constructor(*constructor_args).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.address = tx_receipt['contractAddress']

        if not self.address is None:
            self.address = Web3.to_checksum_address(self.address)
            self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

    def name(self) -> str:
        return ""

    @property
    def eth_balance(self):
        return self.w3.eth.get_balance(self.account.address)

    def _get_params(self) -> Dict:
        params = {
            "chainId": self.w3.eth.chain_id,
            "from": self.account.address,
            "gasPrice": self.w3.eth.gas_price,
            "nonce": self.w3.eth.get_transaction_count(self.account.address, 'pending')
        }

        # gasPrice + 10% to address replacement transaction underpriced error
        if 'sepolia' in self.provider.name():
            increased_gas_price = self.w3.eth.gas_price * 1.1
            params.update({"gasPrice": int(increased_gas_price)})

        if Global.anvil_config_path:
            anvil = Anvil(Global.anvil_config_path)
            params.update({
                "gasPrice": int(anvil.gas_price),
                "gas": 2_000_000
            })

        return params

    def wait_for_transaction(self, tx_hash: HexBytes, raise_on_error: bool = True) -> TxReceipt:

        print(f"\n"
            f'[*] Waiting for transaction...\n'
            f'  > transaction hash: {tx_hash.hex()}'
        )
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if raise_on_error and receipt['status'] == 0:
            self.check_transaction(tx_hash)
        print(f'  > done')
        return receipt

    def send_transact_function(self, func_name: str, *args, log=True, max_retries=5, num_retry=0) -> HexBytes:
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args)

        try:
            tx_params = self._get_params()

            transaction = contract_function.build_transaction(tx_params)
            signed_transaction = self.w3.eth.account.sign_transaction(
                transaction, self.account.key.hex()
            )
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_transaction.rawTransaction
            )

            if log:
                print(f'\n'
                    f'[*] Executing transact function "{func_name}" from "{self.name}"\n'
                    f'  > transaction hash: {Web3.to_hex(tx_hash)}')

            return tx_hash

        except ValueError as err:
            if max_retries < num_retry:
                raise Fatal(f"Max retries raised err={err}")
            d = err.args[0]
            if d['code'] == -32000: 
                num_retry += 1
                print(f"Bypassing Error: {d['message']}")
                print(f"  > Retrying {func_name} {num_retry}/{max_retries}")
                return self.send_transact_function(func_name, *args, log=log, num_retry=num_retry)
            raise ValueError(err)


    def send_view_function(self, func_name: str, *args, log=True):
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args)
        if log:
            print(f'[*] Executing view function "{func_name}" from "{self.name}"')
        return contract_function.call()

    def check_transaction(self, tx_hash):
        tx = cast(Dict, self.w3.eth.get_transaction(tx_hash))
        replay_tx: TxParams = {
            'to': tx['to'],
            'from': tx['from'],
            'value': tx['value'],
            'data': tx['input'],
        }
        try:
            self.w3.eth.call(replay_tx, tx['blockNumber'])
        except ContractLogicError as err:
            raise ContractError(err)


class TokenContract(Contract):
    def __init__(
        self, 
        account, 
        address: Optional[str] = None,
        abi_path: str = Global.token_abi_path,
        bytecode: Optional[Bytecode] = None, 
        abi: Optional[Abi] = None,
        provider: Optional[Provider] = None,
        allow_sign_and_send: bool = False,
    ):
        provider = provider or Provider.build()
        address = address or provider.token_address or Global.token_address
            
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=address, 
            abi_path=abi_path, 
            provider=provider,
            allow_sign_and_send=allow_sign_and_send
        )

    @property
    def name(self) -> str:
        return "TokenContract"

    @property
    def fwai_balance(self):
        return self.get_balance_of(self.account.address)

    def approve(self, address, amount) -> HexBytes:
        return self.send_transact_function("approve", address, amount)

    def get_balance_of(self, address) -> float:
        return self.send_view_function("balanceOf", address)

    def initialize(self) -> HexBytes:
        return self.send_transact_function("initialize")

    def mint(self, address, amount) -> HexBytes:
        return self.send_transact_function("mint", address, amount)

    def burn(self, address, amount) -> HexBytes:
        return self.send_transact_function("burn", address, amount)


class TaskRunnerContract(Contract):
    def __init__(
        self, 
        account, 
        token_address: str = Global.token_address,
        address: Optional[str] = None,
        abi_path: str = Global.task_runner_abi_path,
        provider: Optional[Provider] = None,
        abi: Optional[Abi] = None,
        bytecode: Optional[Bytecode] = None, 
        token_contract: Optional[TokenContract] = None,
        allow_sign_and_send: bool = False,
    ):
        provider = provider or Provider.build()
        address = address or provider.task_runner_address or Global.task_runner_address
        if not token_contract:
            token_address = token_address or provider.token_address or Global.token_address
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=address, 
            abi_path=abi_path, 
            constructor_args=(token_address,),
            provider=provider,
            allow_sign_and_send=allow_sign_and_send,
        )
        self.token = token_contract or TokenContract(
            account, address=token_address, provider=provider
        )
        self.task_price = self.contract.functions.taskPrice().call()

        ## TODO: Decomment this after set public stakingMinimum in contract
        # self.staking_minimum = self.contract.functions.stakingMinimum().call()
        self.staking_minimum = 100

    @property
    def name(self) -> str:
        return "TaskRunnerContract"

    def check_if_ready_to_validate(self, task_id: int, log: bool = True):
        return self.send_view_function('checkIfReadyToValidate', task_id, log=log)

    def get_staking_amount(self, address, log=True):
        return self.send_view_function("stakingAmounts", address, log=False)

    def is_in_timeout(self, task_id: int, log: bool = True) -> int:
        return self.send_view_function('isInTimeout', task_id, log=log)

    def get_available_task_results(
        self, 
        task_id: int, 
        log: bool = True
    ) -> List[ContractNodeResult]:
        results: List[Tuple[str, str, int]] = self.send_view_function(
            'getAvailableTaskResults', task_id, log=log)
        return list(map(lambda tup: ContractNodeResult(*tup), results))

    def add_task(
        self, model_url: str,
        dataset_url: str,
        min_time: int = 1,
        max_time: int = 200,
        min_results: int = 2,
    ) -> Tuple[HexBytes, AttributeDict]:
        account_balance = self.token.get_balance_of(self.account.address)

        if account_balance < self.task_price: 
            raise UserRequirement(
                f'Not enough FWAI tokens to add task.\n'
                f"  > Your balance (FWAI): {account_balance}\n"
                f"  > Price of task: {self.task_price}\n"
                f'Please buy {self.task_price - account_balance} FWAI as minimum'
            )

        tx_hash = self.token.approve(self.address, self.task_price)
        print(f"[DEBUG] in add task {tx_hash=}")
        receipt = self.wait_for_transaction(tx_hash, True)

        tx_hash = self.send_transact_function(
            'addTask', model_url, dataset_url, min_time, max_time, min_results
        )
        receipt = self.wait_for_transaction(tx_hash, True)

        logging.debug(f'{tx_hash=}, {self.task_price=}')
        
        try:
            event_data = self.contract.events.TaskAdded().process_receipt(receipt)[0]
        except IndexError as err:
            raise IndexError(err)

        print(f"\n"
            f"[*] Task added\n"
            f"  > model_url: {model_url}\n"
            f"  > dataset_url: {dataset_url}\n"
            f"  > Transaction hash: {Web3.to_hex(tx_hash)}"
        )
        return tx_hash, event_data

    def get_avaialble_task_results_count(self, task_id) -> int:
        return self.send_view_function('getAvailableTaskResultsCount', task_id)

    def validate_all_tasks_if_ready(self) -> HexBytes:
        return self.send_transact_function('validateAllTasksIfReady')

    def validate_task_if_ready(self, task_id: int) -> HexBytes:
        return self.send_transact_function('validateTaskIfReady', task_id)

    def check_staking_enough(self):
        return self.send_view_function("checkStakingEnough")

    def is_validated(self, task_id, log=True) -> bool:
        return self.send_view_function("isValidated", task_id, log=log)

    def get_tasks_count(self) -> int:
        available_tasks_count = self.send_view_function('getAvailableTasksCount')
        logging.debug(f'[*] Available tasks count = {available_tasks_count}')
        return available_tasks_count

    def get_task_result(self, model_url, dataset_url) -> str:
        return self.send_view_function('getTaskResult', model_url, dataset_url)

    def get_task_time_left(self, task_id) -> int:
        return self.send_view_function('getTaskTimeLeft', task_id)

    def submit_result(self, task) -> HexBytes:
        tx_hash = self.send_transact_function(
            'submitResult', 
            task.id, task.model_url, task.dataset_url, task.result_url
        )
        print(task)
        print(f"\n"
            f"[*] Result Submitted\n"
            f"  > Transaction hash: {Web3.to_hex(tx_hash)}"
        )
        return tx_hash

    def generate_block(self):
        """Testing func"""
        self.send_transact_function('validateAllTasksIfReady', log=False)

    def stake(self, amount) -> HexBytes:
        return self.send_transact_function("stake", amount)

    def unstake(self, amount) -> HexBytes:
        return self.send_transact_function("unstake", amount)


### TODO: !!! Remove it
def testing_mint_to_node(
    node_address, 
    master_private_key: Optional[str] = None,
    amount: int = 200,
):
    f"""
    Testing:
        Mint {amount} fwai to any nodes
    """
    config_path = Global.anvil_config_path
    assert master_private_key or config_path

    if master_private_key is None:
        ## If None, Take the first private_key from anvil
        master_private_key = get_private_key_by_id(1, config_path, "config")

    master_account = get_account(master_private_key)
    token = TokenContract(master_account)
    try:
        token.initialize()
    except:
        ...
    if token.get_balance_of(node_address) < amount:
        token.mint(node_address, amount)
        print(f'[*] Minted {amount} FWAI tokens for this demo')
