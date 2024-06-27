import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch import optim

import data_reader as reader
import itracker_model

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="the file of configuration")
    config = parser.parse_args()
    config = config["train"]
    path = config["data"]["path"]
    model_name = config["save"]["model_name"]

    save_path = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")
    _, dataset = reader.txtload(path, "train", config["params"]["batch_size"], shuffle=True,
                                num_workers=8, device="phone", iTracker_format=True)

    print("Model building")
    net = itracker_model.ITrackerModel()

    net.train()
    net = nn.DataParallel(net)
    # state_dict = torch.load("checkpoint/Iter_10_AFF-Net.pt")
    # net.load_state_dict(state_dict)
    net.to(device)

    print("optimizer building")
    lossfunc = config["params"]["loss"]
    loss_op = getattr(nn, lossfunc)().cuda()
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Traning")
    length = len(dataset)
    total = length * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            for i, data in enumerate(dataset):

                # Acquire data
                data["faceImg"] = data["faceImg"].to(device)
                data["leftEyeImg"] = data["leftEyeImg"].to(device)
                data['rightEyeImg'] = data['rightEyeImg'].to(device)
                data['grid'] = data['grid'].to(device)
                label = data['label'].to(device)
                # label = label.to(device)

                # forward
                gaze = net(data)

                # loss calculation
                loss = loss_op(gaze, label)
                optimizer.zero_grad()

                # backward
                loss.backward()
                optimizer.step()
                scheduler.step()
                cur += 1

                # print logs
                if i % 20 == 0:
                    timeend = time.time()
                    resttime = (timeend - timebegin) / cur * (total - cur) / 3600
                    log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))
