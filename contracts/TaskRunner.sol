// SPDX-License-Identifier: OTHER
pragma solidity ^0.8.9;

import "contracts/Utils.sol";
import "contracts/FreeWillAIToken.sol";

struct Task {
    string model_url;
    string dataset_url;
    string result_url;
    mapping(bytes32 => uint) resultCounts;
    Result[] results;
    uint256 startTime;
    uint reward;
    address sender;
    uint minTime;
    uint maxTime;
    uint minResults;
}

struct Result {
    string result_url;
    address node;
    uint stake;
}

contract TaskRunner {

    FreeWillAI public token;
    Utils utils = new Utils();
    Task[] public availableTasks;
    mapping(bytes32 => string) public resultsMap;
    mapping (address => uint) public stakingAmounts;
    uint stakingMinimum = 100;
    uint public taskPrice = 10;
    uint consensusThreshold = 5; // 1 = 10%, 2 = 20%

    constructor(address _address){
        token = FreeWillAI(_address);
    }
    
    /*
    @param model_url : IPFS url where the AI model are
    @param dataset_url : IPFS url where the dataset are
    @param minTime : Minimum time to wait for results
    @param maxTime : Maximum time to wait for results
    @param minResults : Minimum required results. This is proportional to the security level of the run
    */
    function addTask(string memory model_url, string memory dataset_url, uint minTime, uint maxTime, uint minResults) public {
        require(token.balanceOf(msg.sender) >= taskPrice, "Not enough FWAI tokens to add task.");
        require(minTime < maxTime, "Bad arguments: minTime must be less than maxTime");
        require(1 < minResults, "Bad arguments: minResults must be more than 1");
        token.transferFrom(msg.sender, address(this), taskPrice);

        Task storage task = availableTasks.push();
        task.model_url = model_url;
        task.dataset_url = dataset_url;
        task.startTime = block.timestamp;
        task.reward = taskPrice;
        task.sender = msg.sender;
        task.minTime = minTime;
        task.maxTime = maxTime;
        task.minResults = minResults;

        emit TaskAdded(availableTasks.length - 1, model_url, dataset_url);
    }
    event TaskAdded(
        uint indexed taskIndex,
        string model_url,
        string dataset_url
    );

    function isValidated(uint taskIndex) public view returns (bool) {
        return !(utils.equalStrings(availableTasks[taskIndex].result_url, ""));
    }

    function getAvailableTasksCount() public view returns (uint) {
        return availableTasks.length;
    }

    function getAvailableTaskResults(uint taskIndex) public view returns (Result[] memory) {
        return availableTasks[taskIndex].results;
    }
        
    function getAvailableTaskResultsCount(uint taskIndex) public view returns (uint) {
        return availableTasks[taskIndex].results.length;
    }

    function getAvailableTask(uint taskIndex) public view returns (string memory, string memory) {
        require(taskIndex < availableTasks.length, "Invalid index");
        Task storage task = availableTasks[taskIndex];
        return (task.model_url, task.dataset_url);
    }

    function submitResult(uint taskIndex, string calldata model_url, string calldata dataset_url, string calldata result_url) public {
        require(taskIndex < availableTasks.length, "Task doesn't exist. taskIndex too high");
        require(!isInTimeout(taskIndex), "Submitting outside of this task's time window. Too late");
        require(checkStakingEnough(), "Your stake is not high enough to submit a result");
        Task storage task = availableTasks[taskIndex];
        require(utils.equalStrings(task.model_url, model_url), "model_url doesn't match the task on that index");
        require(utils.equalStrings(task.dataset_url, dataset_url), "dataset_url doesn't match the task on that index");
        
        Result memory result = Result(result_url, msg.sender, stakingMinimum);
        stakingAmounts[msg.sender] -= stakingMinimum;
        
        task.results.push(result);
    }

    function validateAllTasksIfReady() public{
        for(uint i = 0; i<availableTasks.length; i++){
            validateTaskIfReady(i);
        }
    }
    
    function stake(uint amount) public{
        token.transferFrom(msg.sender, address(this), amount);
        stakingAmounts[msg.sender] += amount;
    }

    function unstake(uint amount) public{
        require(stakingAmounts[msg.sender] >= amount, "Not enough tokens staked");
        token.approve(address(this), amount);
        token.transfer(msg.sender, amount);
        stakingAmounts[msg.sender] -= amount;
    }

    function getTaskResult(string calldata model_url, string calldata dataset_url) public view returns (string memory){
        bytes32 taskHash = utils.hash2(model_url, dataset_url);
        return resultsMap[taskHash];
    }

    function checkStakingEnough() view public returns (bool){
        // require(msg.sender.balance > stakingMinimum);
        return stakingAmounts[msg.sender] >= stakingMinimum;
    }

    function isInTimeout(uint taskIndex) view public returns (bool){
        Task storage task = availableTasks[taskIndex];
        return (block.timestamp > task.startTime + task.maxTime);
    }

    function checkIfReadyToValidate(uint taskIndex) view public returns (bool){
        Task storage task = availableTasks[taskIndex];
        return (
            task.startTime + task.minTime <= block.timestamp 
            && block.timestamp <= task.startTime + task.maxTime 
            && task.minResults <= getAvailableTaskResultsCount(taskIndex)
        );
    }

    function getTaskTimeLeft(uint taskIndex) view public returns (int){
        Task storage task = availableTasks[taskIndex];
        return int(task.startTime + task.maxTime) - int(block.timestamp);
    }
    function getTimestamp() view public returns (uint){
        return block.timestamp;
    }
    function getblocktime() private pure returns (uint256 result){
        return 0 ;
    }

    function validateTaskIfReady(uint taskIndex) public{
        Task storage task = availableTasks[taskIndex];
        getblocktime();
        if(checkIfReadyToValidate(taskIndex) && utils.isEmptyString(task.result_url)){
            task.result_url = getValidResult(taskIndex);
            bytes32 taskHash = utils.hash2(task.model_url, task.dataset_url);
            resultsMap[taskHash] = task.result_url;
            rewardAndPunishNodes(taskIndex);
        } else if (isInTimeout(taskIndex)) {
            // Payment return to user
            token.approve(address(this), taskPrice);
            token.transfer(task.sender, taskPrice);
        }

    }
    

    function getValidResult(uint taskIndex) internal returns (string memory){
        Task storage task = availableTasks[taskIndex];
        Result[] memory results = task.results;
        string memory mostPopularResult;
        uint mostPopularResultCount = 0;
        for(uint i = 0; i < results.length; i++){
            string memory result_url = results[i].result_url;
            bytes32 result_url_hash = utils.hash(result_url);
            task.resultCounts[result_url_hash] += 1;

            if(task.resultCounts[result_url_hash] > mostPopularResultCount){
                mostPopularResult = result_url;
                mostPopularResultCount = task.resultCounts[result_url_hash];
            }
        }
        return mostPopularResult;
    }
    
    function rewardAndPunishNodes(uint taskIndex) internal{
        Task storage task = availableTasks[taskIndex];
        Result[] memory results = task.results;
        uint totalStake = 0;
        uint totalCorrect = 0;
        for(uint i = 0; i < results.length; i++){
            totalStake += results[i].stake;
            if(utils.equalStrings(results[i].result_url, task.result_url)){
                totalCorrect++;
            }
        }
        uint consensusNeeded = (results.length * consensusThreshold) / 10;
        if(totalCorrect > consensusNeeded) {
            uint totalReward = totalStake + task.reward;
            for(uint i = 0; i < results.length; i++){
                if(utils.equalStrings(results[i].result_url, task.result_url)){
                    stakingAmounts[results[i].node] += totalReward / totalCorrect;
                }
            }
        }
        else {
            // Return stake to nodes
            for(uint i = 0; i < results.length; i++){
                stakingAmounts[results[i].node] += stakingMinimum;
            }
            // Payment return to user
            token.approve(address(this), taskPrice);
            token.transfer(task.sender, taskPrice);
       }
    }
}
    
