from freewillai.core import run_task, connect


__doc__ = """
Example
-------
You have two ways to create a new project:

1:
>>> import freewillai
>>> my_project = freewillai.create_project('my_project')

2:
>>> from freewillai import Project
>>> my_project = Project('my_project')

Adding and running two tasks on freewillai nodes:

>>> my_project.add_task(model_1, dataest_1)
>>> my_project.add_task(model_2, dataset_2)
>>> my_project.pay_and_run()

Adding a Instanced tasks

>>> task_3 = my_project.create_task(model_3, dataset_3)
>>> task_4 = my_project.create_task(model_4, dataset_4)
>>> task_5 = my_project.create_task(model_5, dataset_5)

The instanced tasks can be modified

>>> task_3.model = torch.load('path/to/pytorch_model.pt')
>>> task_5.dataset = np.array([1,2], [3,4], [5,6])

Running just the selected tasks

>>> my_project.pay_and_run(task_3, task_5)
"""


__all__ = ['run_task', 'connect']
