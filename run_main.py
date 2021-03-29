import argparse
from train import Train, TEST, run


parser = argparse.ArgumentParser(description="Train Test Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--model", default="SE_resnet", type=str, dest="model")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--csv_dir", default="./csv", type=str, dest="csv_dir")
parser.add_argument("--num_epoch", default=15, type=int, dest="num_epoch")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train" or "test":
        run(args)
    else:
        print("Error: Check mode: you can choose between train or test")
