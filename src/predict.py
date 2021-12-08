import torch
import torch.nn as nn
import numpy as np
from piq import psnr, ssim
import time
import cv2

class Predicter():
    def __init__(self, model, test_loader, save_path):
        self.model = model
        self.test_loader = test_loader
        self.save_path = save_path
        self.test_loss=[]
        self.psnr=[]
        self.ssim=[]

    def predict(self, args):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        self.model.to(device)

        self.model.eval()

        test_loss = 0
        test_psnr = 0
        test_ssim = 0
        for batch_input_x, batch_input_y, name in self.test_loader:
            batch_input_x = batch_input_x.to(device)
            batch_input_y = batch_input_y.to(device)

            with torch.no_grad():
                output = self.model(batch_input_x)

            loss = self.loss_function(output, batch_input_y)
            test_loss += loss.item()
            
            metric_psnr = psnr(output, batch_input_y, data_range=1.)
            metric_ssim = ssim(output, batch_input_y, data_range=1.)
            
            test_psnr += metric_psnr.item()
            test_ssim += metric_ssim.item()

            output_images = output.cpu().detach().numpy()
            output_images = (output_images * 255.0).transpose(0,2,3,1).astype(np.uint8)  # b, c, h, w -> b, h, w, c

            for img in output_images:
                cv2.imwrite(os.path.join(save_path,name),img)


        print('test_loss: ', test_loss/len(self.test_loader))
        print('test_psnr: ', test_psnr/len(self.test_loader)) 
        print('test_ssim: ', test_ssim/len(self.test_loader))