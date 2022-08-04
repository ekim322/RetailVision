import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import Alc_Dataset
from TripletNet import TripletNet, TripletMiningLoss

# parser = argparse.ArgumentParser(description='3dUnet Training')
# parser.add_argument('--config', default='train_config.yaml', type=str)

class Args():
    def __init__(self):
        self.config = 'train_config.yaml'
args = Args()

def run_epoch(model, optimizer, data_loader, epoch, data_type, device):
    par_loss_fn = TripletMiningLoss()
    par_cls_loss_fn = torch.nn.CrossEntropyLoss()
    sub_loss_fn = TripletMiningLoss()
    running_loss = 0
    running_par_loss = 0
    # running_par_cls_loss = 0
    running_sub_loss = 0
    # num_correct = 0
    # num_data = 0

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (imgs, par_labels, sub_labels) in data_loop:
        imgs = imgs.to(device)
        par_labels = par_labels.to(device)
        sub_labels = sub_labels.to(device)
        
        embs = model(imgs)
        
        par_loss = par_loss_fn(embs, par_labels)
        # par_cls_loss = par_cls_loss_fn(cls_preds, par_labels)
        sub_loss = sub_loss_fn(embs, sub_labels)
        # loss = par_loss + sub_loss #+ par_cls_loss
        loss = sub_loss #+ par_cls_loss

        running_par_loss += par_loss.item()
        # running_par_cls_loss += par_cls_loss.item()
        running_sub_loss += sub_loss.item()
        running_loss += loss.item()

        if data_type == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # _, predictions = torch.max(cls_preds.data, 1)
        # num_correct += (predictions==par_labels).sum().item()
        # num_data += len(predictions)

        loop_description = "{} epoch {}".format(data_type, epoch)
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss.item())

    epoch_par_loss = running_par_loss / (i + 1)
    # epoch_par_cls_loss = running_par_cls_loss / (i + 1)
    epoch_sub_loss = running_sub_loss / (i + 1)
    epoch_loss = running_loss / (i + 1)
    # cls_acc = num_correct / num_data

    return epoch_par_loss, epoch_sub_loss, epoch_loss#, epoch_par_cls_loss, cls_acc

def train_model(model, cfg, device):
    if not os.path.exists(cfg["model_save_root"]):
        os.makedirs(cfg["model_save_root"])
    VAL_SAVE_PATH = os.path.join(cfg["model_save_root"], cfg["exp_name"]+".pt")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, cfg["lr"])

    train_dataset = Alc_Dataset(
        img_dir=cfg["train_dir"],
        img_size=cfg["img_size"]
    )

    # num_val = int(len(train_dataset)*0.1)
    # num_train = len(train_dataset) - num_val
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
    # print("num train {}, num_val {}".format(num_train, num_val))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    val_dataset = Alc_Dataset(
        img_dir=cfg["val_dir"],
        img_size=cfg["img_size"]
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    writer = SummaryWriter(os.path.join('logs', cfg['exp_name'])) 
    best_val_loss = 1000

    for epoch in range(cfg["epochs"]):
        model.train()
        train_par_loss, train_sub_loss, train_loss = run_epoch(model, optimizer, train_dataloader, epoch, "train", device)
        writer.add_scalar("loss/train_par_loss", train_par_loss, epoch)
        # writer.add_scalar("loss/train_par_cls_loss", train_par_cls_loss, epoch)
        writer.add_scalar("loss/train_sub_loss", train_sub_loss, epoch)
        writer.add_scalar("loss/train_loss", train_loss, epoch)
        # writer.add_scalar("acc/train", train_cls_acc, epoch)

        model.eval()
        val_par_loss, val_sub_loss, val_loss = run_epoch(model, None, val_dataloader, epoch, "val", device)
        writer.add_scalar("loss/val_par_loss", val_par_loss, epoch)
        writer.add_scalar("loss/val_sub_loss", val_sub_loss, epoch)
        writer.add_scalar("loss/val_loss", val_loss, epoch)

        print("Epoch {} - Train Loss: {} - Val Loss: {}"
              .format(epoch, train_loss, val_loss))

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), VAL_SAVE_PATH)
            
if __name__=='__main__':
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TripletNet()
    model.to(device)

    train_model(model, cfg, device)
