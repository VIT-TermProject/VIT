import torch
import torch.nn as nn
import numpy as np
from piq import psnr, ssim
import time

class Trainer():
    def __init__(self, model, train_loader, valid_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_loss=[]
        self.valid_loss=[]
        self.psnr=[]
        self.ssim=[]

    def train(self, args):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        
        if args.loss_type == 'MSE':
            self.loss_function = nn.MSELoss()
        elif args.loss_type == 'L1':
            self.loss_function = nn.L1Loss()

        for epoch in range(args.epochs):
            start_time = time.time()
            self.model.train()

            train_loss = 0
            for batch_input_x, batch_input_y in self.train_loader:
                batch_input_x = batch_input_x.to(device)
                batch_input_y = batch_input_y.to(device)
                self.model.zero_grad()

                output = self.model(batch_input_x)
                loss = self.loss_function(output, batch_input_y)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            ## validation
            self.model.eval()
            val_loss = 0
            val_psnr = 0
            val_ssim = 0
            for batch_input_x, batch_input_y in self.valid_loader:
                batch_input_x = batch_input_x.to(device)
                batch_input_y = batch_input_y.to(device)

                with torch.no_grad():
                    output = self.model(batch_input_x)

                loss = self.loss_function(output, batch_input_y)
                val_loss += loss.item()
                
                metric_psnr = psnr(output, batch_input_y, data_range=1.)
                metric_ssim = ssim(output, batch_input_y, data_range=1.)
                
                val_psnr += metric_psnr.item()
                val_ssim += metric_ssim.item()

            print('epoch: ', epoch, ', time: ', time.time()-start_time)
            print('train_loss: ', train_loss/len(self.train_loader), 'val_loss: ', val_loss/len(self.valid_loader))
            print('val_psnr: ', np.mean(val_psnr), ', val_ssim: ', np.mean(val_ssim))

            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, args.checkpoint_path + epoch + '.pth')