import asyncio
import torch
from torch import nn
from torch.nn import functional as F
import freewillai

from torchvision import transforms
import numpy as np
from PIL import Image

from freewillai.globals import Global


async def torch_demo():
    print('\n\n[*] Dispatching torch model with image_path...')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    model = Net()
    state_dict = torch.load('bucket/test/models/cifar_net.pth')
    model.load_state_dict(state_dict)

    image_path = 'bucket/test/datasets/cat.png'
    # image = Image.open(image_path)
    # image = image.convert('RGB')

    # transform = transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.ToTensor()
    # ])

    # image = transform(image)

    freewillai.connect(Global.provider_endpoint)
    result = await freewillai.run_task(model, image_path)
    return result


async def keras_demo():
    print('\n\n[*] Dispatching keras model with csv...')
    import keras

    model_path = 'bucket/test/models/keras_model_dnn/'
    model = keras.models.load_model(model_path)
    dataset = 'bucket/test/datasets/keras_testing_dataset.csv'

    freewillai.connect(Global.provider_endpoint)
    result = await freewillai.run_task(model, dataset)
    return result


async def sklearn_demo():
    print('\n\n[*] Dispatching sklearn model with csv...')
    import pickle
    model_path = 'bucket/test/models/sklearn_model.pkl'
    model =  pickle.load(open(model_path, "rb"))
    dataset = 'bucket/test/datasets/dummy_data_set_sklearn.csv'

    freewillai.connect(Global.provider_endpoint)
    result = await freewillai.run_task(model, dataset)
    return result


if __name__ ==  '__main__':
    from freewillai.contract import testing_mint_to_node
    from freewillai.utils import get_account

    account = get_account()
    testing_mint_to_node(account.address)

    asyncio.run(torch_demo())
    asyncio.run(sklearn_demo())
    asyncio.run(keras_demo())
