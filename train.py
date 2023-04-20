import torch
import cv2
import numpy as np
import time
import asyncio

import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from EVSRNET import EVSRNet

from torchvision.transforms import ToTensor, ToPILImage


os.makedirs("./Checkpoints/", exist_ok=True)

print("cuda_is_available : ", torch.cuda.is_available())


class ImageSRTrainer:
    def __init__(self, model_name, scale, num_epochs=50, batch_size=1):
        self.model_name = model_name
        self.scale = scale
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.model = EVSRNet()
        self.model.load_state_dict(torch.load('C:/Users/MOBED/Omni_Ingest/Checkpoints/model.pth'))
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device : ", self.device)
        
        self.T2T = ToTensor() # numpy or PIL_Image to Tensor
        self.T2P = ToPILImage() # Tensor to PIL_Image

    async def train_w_img(self, model, data):
        test_hr = data
        test_hr = cv2.cvtColor(test_hr, cv2.COLOR_BGR2RGB)
        test_hr = cv2.resize(test_hr, (480*4, 270*4), interpolation=cv2.INTER_CUBIC)
        test_hr = test_hr.astype(np.float32)
        test_hr = test_hr / 255.
        test_lr = cv2.resize(test_hr, (480, 270), interpolation=cv2.INTER_CUBIC)
        test_lr = torch.from_numpy(np.transpose(test_lr, (2, 0, 1))).float()
        test_hr = torch.from_numpy(np.transpose(test_hr, (2, 0, 1))).float()
        test_lr = torch.unsqueeze(test_lr, 0)
        test_hr = torch.unsqueeze(test_hr, 0)

        before_loss = 100

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        model.to(self.device)

        for epoch in tqdm(range(self.num_epochs)):
            train_loss = 0
            model.train()

            test_lr, test_hr = test_lr.to(self.device), test_hr.to(self.device) # data = test_lr, target = test_hr
            # print('data shape : ',data.shape) # (N x C X H X W)
            # print('data : ',data)
            # print('target shape : ',target.shape) # (N x C X H X W)
            # print('target : ',target)
            optimizer.zero_grad()
            output = model(test_lr)
            loss = criterion(output, test_hr) # criterion(data, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if ((train_loss/self.num_epochs) <= before_loss):
            print('Training Loss: {:.8f}'.format(train_loss/self.num_epochs))
            before_loss = train_loss/self.num_epochs
            torch.save(model.state_dict(), "./Checkpoints/{}.pth".format(self.model_name))
        # output = torch.squeeze(output,0) # tensor(N x C x H x W) -> numpy(C x H x W)
        # output = T2P(output)
        # test_hr = torch.squeeze(test_hr,0) # tensor(N x C x H x W) -> numpy(C x H x W)
        # test_hr = T2P(test_hr)
        # plt.imshow(output)
        # plt.show()
        # plt.imshow(test_hr)
        # plt.show()
        
    async def train (self, model, data):
        start = time.time()
        await self.train_w_img(model, data)
        end = time.time()
        print(f"Time taken: {(end-start):.6f} seconds")
