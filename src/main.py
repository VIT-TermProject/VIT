import os
import argparse
import torch
from model import ViT
from train import train
from dataloader import PlacesDataset
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="")
    parser.add_argument("--valid_path", default="")
    parser.add_argument("--test_path", default="")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--loss_type", type=str, default="L1")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_path", type=str, default="")

    args = parser.parse_args()

    train_dataset = PlacesDataset(args.train_path, normalize=False, augmentation=False)
    valid_dataset = PlacesDataset(args.valid_path, normalize=False, augmentation=False)
    test_dataset = PlacesDataset(args.test_path, normalize=False, augmentation=False)

    model = ViT(args, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
    
    if args.mode == 'train':
        train_loader = DataLoader(train_dataset, args.batch_size)
        valid_loader = DataLoader(valid_dataset, args.batch_size)

    elif args.mode == 'test':
        test_loader = DataLoader(test_dataset, args.batch_size)

    train(args, model, train_loader, valid_loader)

if __name__ == "__main__":
    main()