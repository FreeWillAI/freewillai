import os
import asyncio
import random
import numpy as np
import zipfile
import time
import argparse
from typing import List
from tempfile import NamedTemporaryFile
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer 
from freewillai.common import Anvil, IPFSBucket, Provider, SepoliaProvider, TestDataset, OnnxModel, Task
from freewillai.globals import Global
from freewillai.utils import get_account, get_url, in_cache, get_path, generate_text, load_global_env
from freewillai.contract import TaskRunnerContract, TokenContract
from freewillai.exceptions import Fatal
from dotenv import load_dotenv
from typing import Optional
from web3.exceptions import MethodUnavailable


class Node:

    def __init__(
        self,
        # TODO: Multiprocessing
        cores: int = 1,
        private_key: Optional[str] = None,
        test_bad_result: bool = False,
        test_bad_chance: Optional[float] = None,
        anvil: Optional[Anvil] = None,
        provider: Optional[Provider] = None,
        cooling_time: float = 0,
    ):
        self.private_key = private_key
        self.test_bad_result = test_bad_result
        self.test_bad_chance = test_bad_chance

        self.provider = provider or Provider.build()

        self.account = get_account(self.private_key)
        self.provider.allow_sign_and_send(self.account)

        self.address = self.account.address
        self.token = TokenContract(self.account, provider=self.provider)
        self.task_runner = TaskRunnerContract(
            self.account, 
            provider=self.provider, 
            token_contract=self.token
        )
        self.loop = asyncio.get_event_loop()
        self.anvil = anvil

        self.pending_tasks: List[int] = []
        self.cooling_time = cooling_time

        if os.environ.get('DEMO'):
            self.__testing_minting_200_fwai()

        if not self.token.fwai_balance > 0:
            raise Fatal(
                ## TODO: Use self.task_runner.staking_minimum instead of harcoded 100 
                ## after deploy task runner contract on sepolia
                f"You have {self.token.fwai_balance} FWAI in your wallet. "
                f"Please buy FWAI enough to stake (100 FWAI at least)"
            )

        self.staking_amount: int = self.task_runner.get_staking_amount(self.address)

        print(f'\n[*] Account Info')
        print(f'  > Your public key is {self.address}')
        print(f'  > ETH in account: {self.token.eth_balance}')
        print(f'  > FWAI in account: {self.token.fwai_balance}')

        
    ### TODO: !!! Remove it
    def __testing_minting_200_fwai(self, amount=200):
        from freewillai.contract import testing_mint_to_node
        testing_mint_to_node(self.address, amount=amount)


    def stake(self, amount):
        self.staking_amount = self.task_runner.get_staking_amount(self.address)
        
        if amount > self.token.fwai_balance:
            raise Fatal(
                f"Staking amount ({amount} FWAI) "
                f"exceeds your balance ({self.token.fwai_balance} FWAI). "
                f"Please buy FWAI to stake"
            )

        if self.staking_amount < amount:
            tx_hash = self.token.approve(self.task_runner.address, amount)
            self.token.wait_for_transaction(tx_hash)
            tx_hash = self.task_runner.stake(amount - self.staking_amount)
            # This is just for print
            self.token.wait_for_transaction(tx_hash)

        elif self.staking_amount > amount:
            tx_hash = self.task_runner.unstake(self.staking_amount - amount)
            # This is just for print
            self.task_runner.wait_for_transaction(tx_hash)
            
        self.staking_amount = self.task_runner.get_staking_amount(self.address)
        print(f'  > Staking amount: {self.staking_amount}')


    async def run_task(self, task):
        if not in_cache(task.model_url):
            print(f"[*] Downloading {task.model_url}")
            await IPFSBucket.download(task.model_url, file_type='model')
        if not in_cache(task.dataset_url):
            print(f"[*] Downloading {task.dataset_url}")
            await IPFSBucket.download(task.dataset_url, file_type='dataset')
        print('\n[*] Running Inference')

        dataset = TestDataset(task.dataset_url)
        model_path = get_path(task.model_url)
        is_huggingface = False
        if not os.path.exists(model_path):
            print('Did not find the model_path. Sleeping and trying again')
            time.sleep(3)
            is_huggingface = True
        elif os.path.isdir(model_path):
            is_huggingface = True
            print('Found a directory. Sleeping to wait for full unzipping (done by another node)')
            time.sleep(3)
        if zipfile.is_zipfile(model_path):
            try:
                os.rename(model_path, model_path+'.zip')
                with zipfile.ZipFile(model_path+'.zip', 'r') as zip_ref:
                    zip_ref.extractall(model_path)
                    is_huggingface = True
            except Exception as err:
                print("Can't unzip. Sleeping", err)
                time.sleep(3)

        if is_huggingface:
            model = AutoModelForCausalLM.from_pretrained(model_path+'/model')
            tokenizer = AutoTokenizer.from_pretrained(model_path+'/tokenizer')
            text = open(dataset.path, "r").read()
            result = generate_text(model = model, tokenizer = tokenizer, sequence=text, max_length = 12)
            print('Result is: ', result)
             
        else:
            model = OnnxModel(task.model_url)
            result = model.inference(dataset)

        if ((not self.test_bad_chance is None 
            and random.random() < self.test_bad_chance)
            or self.test_bad_result
        ):
            result = np.array([1])

        print('  > Inference Done\n')

        await self.submit_result(task, result)


    async def submit_result(self, task, result):
        ## TODO: choice between csv or numpy to upload a result 
        temp_result_file = NamedTemporaryFile().name # + '.npy'
        # np.save(temp_result_file, result)
        if isinstance(result, str):
            with open(temp_result_file, "w") as resultfile:
                resultfile.write(result)
        else:
            np.savetxt(temp_result_file, result, delimiter=";")
        cid = await IPFSBucket.upload(temp_result_file, file_type='result')
        task.submit_result(get_url(cid), self.task_runner)
        self.pending_tasks.append(task.id)


    def latest_block(self):
        return self.task_runner.w3.eth.block_number

    
    def _event_filter_by_get_logs(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block() - 5
        params = {"fromBlock": from_block, "toBlock": 'latest'}
        logs = self.task_runner.contract.events.TaskAdded().get_logs(params)

        # Doble check to address some provider issues that gets logs before 
        # than block number filtering
        for event in logs:
            if event["blockNumber"] > from_block:
                yield event
    

    def _event_filter_by_create_filter(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block() - 5
        return (
            self.task_runner.contract.events.TaskAdded()
            .create_filter(fromBlock=from_block, toBlock='latest')
        ).get_new_entries()


    def event_filter(self):
        try:
            self._event_filter_by_create_filter()
            return self._event_filter_by_create_filter
        
        except MethodUnavailable:
            # print("[DEBUG] MethodUnavailable error. Listening events with get_logs")
            # This is for nets like goerli that does not allow event filtering
            return self._event_filter_by_get_logs

        except ValueError as err:
            if not err.args[0].get("message") == 'filter not found':
                raise Fatal(err)
            return self._event_filter_by_get_logs

        except Exception as err:
            raise Fatal(err)


    async def listen_for_event(self):
        last_block_scanned = self.latest_block()
        event_filter = self.event_filter()
        import time
        while True:
            # print(f"[DEBUG] {last_block_scanned=} | {self.latest_block()=}")
            # Transforming event_filter to callable so always call get_new_entries 
            # if create_filter is called 
            for event in event_filter(last_block_scanned):
                # Recap block just in case
                last_block_scanned = self.latest_block() - 1

                task = Task.load_from_event(event, self.provider)
                if (task.id in self.pending_tasks
                    or self.task_runner.is_validated(task.id, log=False)
                    or self.task_runner.is_in_timeout(task.id, log=False)
                ):
                    continue

                staking_enough = self.task_runner.get_staking_amount(
                    self.address, log=False
                ) >= self.task_runner.staking_minimum

                if not staking_enough:
                    print("[!] Insufficient staking amount to run found task")
                    break                        

                print(f'\n[*] Task found: {event["args"]}')
                await self.run_task(task)
            
            # Pending tasks handling
            for idx, task_id in enumerate(self.pending_tasks):
                if (self.task_runner.is_in_timeout(task_id, log=False)
                    or self.task_runner.is_validated(task_id, log=False)
                ):
                    self.pending_tasks.pop(idx)

                # Maybe validate if ready.
                # if self.task_runner.check_if_ready_to_validate(pending_task.id):
                #     self.task_runner.validate_task_if_ready(pending_task.id)
            await asyncio.sleep(self.cooling_time)


    def spin_up(self):
        import traceback
        print('\n[*] Spining up the node')
        try:
            self.loop.run_until_complete(
                asyncio.gather(self.listen_for_event())
            )

        except KeyboardInterrupt:
            print(traceback.format_exc(chain=False))

        except Fatal as err:
            raise Fatal(err)

        except Exception as err:
            print('Exception on node.spin_up(): ', err)
            traceback.print_exc()
            return self.spin_up()

        finally:
            print('\n[!] Node Killed')
            self.loop.close()


def cli():
    """Testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--private-key', type=str)
    parser.add_argument('-s', '--stake', type=int)
    parser.add_argument('-b', '--bad-result', action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--id', type=int)
    parser.add_argument('-e', '--env-file', type=str)
    parser.add_argument('-c', '--anvil-config', type=str)
    parser.add_argument('-B', '--bad-chance', type=float)
    parser.add_argument('-r', '--rpc-url', type=str)
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-a', '--api-key', type=str)
    parser.add_argument('--cooling-time', type=str)
    
    args = parser.parse_args()

    assert isinstance(args.stake, int)
    private_key = args.private_key if not args.private_key is None else None
    staking_amount = args.stake or 100
    anvil = None

    # It's valid just for anvil tests
    if args.id and private_key is None:
        assert args.anvil_config or Global.anvil_config_path

        config_path = args.anvil_config or Global.anvil_config_path
        Global.anvil_config_path = config_path
        anvil = Anvil(config_path, build_envfile=True)
        account = getattr(anvil, f"node{args.id}")
        private_key = account.private_key

    if args.env_file:
        load_global_env(args.env_file)
        Global.update()

    if args.rpc_url:
        os.environ["FREEWILLAI_PROVIDER"] = args.rpc_url
        Global.update()

    # TODO: Add middlewares and decomment
    provider = Provider.by_network_name(args.network or "default")

    # At moment api_key is more important than rpc_url
    uri = args.api_key or args.rpc_url

    cooling_time = args.cooling_time or 0

    try:
        provider.asserts(uri)
    except AssertionError as err:
        provider.exception(str(err))

    # If not uri build provider by environment
    provider = provider.build(uri)

    max_retries = 5
    def try_spin_up(num_retry: int):
        if max_retries < num_retry:
            return
        try:
            node = Node(
                private_key=private_key, 
                test_bad_result=args.bad_result, 
                test_bad_chance=args.bad_chance, 
                anvil=anvil,
                provider=provider,
                cooling_time=cooling_time,
            )
            node.stake(staking_amount)
            node.spin_up()

        except Fatal as err:
            raise Fatal(err)

        except Exception:
            import traceback
            err_path = f"/tmp/freewillai-error-in-{num_retry}"
            with open(err_path, "w") as err_file:
                err_file.write(traceback.format_exc(chain=False))
            print(f"[!] Retrying due to an unexpected error in iteration {num_retry}/{max_retries}")
            print(f"  > To see the complete log of the error=`cat {err_path}`\n")
            with open(err_path, 'r') as f:
                print(f.read())
            try_spin_up(num_retry+1)

    try_spin_up(1)


if __name__ == '__main__':
    cli()
