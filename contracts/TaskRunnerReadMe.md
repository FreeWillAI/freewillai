This Solidity contract is a decentralized task runner for coordinating the execution of tasks by multiple nodes in a network. It allows users to add tasks with specific models and datasets, and nodes to submit results for those tasks. The contract then determines the most popular result (i.e., the one submitted by the majority of nodes) and stores it for future reference.

1. Imports and Structs:

  •The contract imports the Utils.sol utility contract for performing common operations like string comparison and hashing.
  
  •It defines two structs, Task and Result, to represent a task and its associated results, respectively.
  
  
  
2. Contract and Variables:

  •The contract is named TaskRunner and contains an instance of the Utils contract.
  
  •It maintains an array of available tasks, a mapping of task hashes to their corresponding most popular results, and a taskTimeWindow variable to specify the time window for task execution.



3. Adding Tasks:

  •The addTask() function allows users to add a new task to the list of available tasks. It initializes the task with the given model and dataset URLs and sets the start time of the task to the current block timestamp.
  
  •An event TaskAdded is emitted when a task is added.



4. Querying Tasks and Results:

  •The contract provides various functions to query the available tasks and their results, including getAvaialbleTasksCount(), getAvaialbleTaskResultsCount(), and getAvailableTask().



5. Submitting Results:

  •The submitResult() function allows nodes to submit a result for a specific task if the task is still within the allowed time window. It checks that the provided model and dataset URLs match the task's data before adding the result to the task's results array.



6. Validating and Storing Results:

  •The validateAllTasksIfReady() function iterates through all tasks and checks if they are ready for validation using validateTaskIfReady(). A task is considered ready for validation if it is no longer within the allowed time window and it does not already have a final result set.
  
  •The getValidResult() function determines the most popular result among the submitted results for a task and returns it.
  
  •The validateTaskIfReady() function sets the most popular result as the final result for a task and adds it to the resultsMap.



7. Time-related Functions:

  •The checkIfWithinTimeWindow() function checks if a task is still within the allowed time window.
  
  •The getTaskTimeLeft() function returns the remaining time for a task to be executed.
  
  •The getTimestamp() function returns the current block timestamp.
  
