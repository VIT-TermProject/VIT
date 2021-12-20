import os
import argparse
import torch
from model import ViT
from model_cnn import My
from train import Trainer
from dataloader import PlacesDataset
from torch.utils.data import DataLoader
from predict import Predicter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="../DATA/noise/train")
    parser.add_argument("--valid_path", default="../DATA/noise/val")
    parser.add_argument("--test_path", default="")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--loss_type", type=str, default="L1")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_path", type=str, default="../resultpatch4r")

    args = parser.parse_args()

    train_dataset = PlacesDataset(args.train_path, data_type='train', normalize=True, augmentation=False)
    valid_dataset = PlacesDataset(args.valid_path, data_type='test', normalize=True, augmentation=False)
    test_dataset = PlacesDataset(args.valid_path,data_type='test', normalize=True, augmentation=False)

    model = ViT(image_size=64,patch_size=16,num_classes=1,dim=512,mlp_dim=1024, depth=6, heads=16)
    #model = My()
    
    if args.mode == 'train':
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        trainer = Trainer(model, train_loader, valid_loader)
        trainer.train(args)

    elif args.mode == 'test':
        model.load_state_dict(torch.load("../resultpatch4/1.pth")['model_state_dict'])
        test_loader = DataLoader(test_dataset, 32)
        predicter = Predicter(model, test_loader, "../resultimg3")
        predicter.predict()

    
if __name__ == "__main__":
    main()