import torch
import torch.nn as nn
from tqdm import tqdm
from utils import Save_Load_Checkpoint, Metrics


class Train:
    def __init__(self,
                 train_dataloader,
                 validation_dataloader,
                 test_dataloader,
                 net,
                 loss_fn,
                 optimizer,
                 config,
                 save_path
                 ) -> None:

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.net = net
        self.checkpoint = Save_Load_Checkpoint(save_path)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.acc = 0
        self.validation_metric = []
        self.test_metric = []
        self.config = config

    def train(self):
        acc = 0
        self.net.train()
        loop = tqdm(self.train_dataloader)
        for epoch in range(self.config.EPOCH):
            print(f"-----Epoch: {epoch + 1} -----")
            loss_total = 0
            data_size = 0
            for idx, (seq, label) in enumerate(loop):
                self.optimizer.zero_grad()
                output = self.net(seq)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                data_size += seq[0]

            print(f"Train_loss is {loss_total / data_size}.")
            print("----- Save CheckPoint -----")
            if epoch % 10 == 0:
                self.validation()
                if self.acc > acc:
                    self.checkpoint.save_checkpoint(self.net, True)
                    acc = self.acc
                else:
                    self.checkpoint.save_checkpoint(self.net, False)

    def validation(self):
        label_list = []
        label_predict = []
        loss_val = 0
        with torch.no_grad():
            for idx_val, (seq_val, label_val) in enumerate(self.validation_dataloader):
                output_val = self.net(seq_val)
                loss = self.loss_fn(output_val, label_val)
                loss_val += loss.item()
                output_val = output_val.argmax(dim=-1)
                label_predict.extend(output_val.tolist())
                label_list.extend(label_val.tolist())

            cal_metric = Metrics(label_list, label_predict)
        self.validation_metric.append(cal_metric.get_all())
        self.acc = cal_metric.get_acc()
        print(
            f"Acc: {cal_metric.get_acc()}. Sn: {cal_metric.get_sn()}. Sp: {cal_metric.get_sp()}. MCC: {cal_metric.get_mcc()}.")



