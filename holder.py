import torch
import torch.optim as op
import torch.nn as nn
import torch.nn.functional as F

import utils
from model import Model


class Holder():
    def __init__(self, args):
        self.args = args
        self.model = Model(args).to(self.args.device)
        self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss = utils.masked_mae
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def train(self, inputs, reals):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        reals = reals[:, :self.args.seq_out, :, :]
        prediction = self.args.scaler.inv_transform(outputs)
        loss = self.loss(prediction, reals, 0.0)
        loss.backward(retain_graph=True)
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
        mape = utils.masked_mape(prediction, reals, 0.0).item()
        rmse = utils.masked_rmse(prediction, reals, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, inputs, reals):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
        reals = reals[:, :self.args.seq_out, :, :][:, -1, :, :]
        # reals = reals[:, :self.args.seq_out, :, :]
        prediction = self.args.scaler.inv_transform(outputs)[:, -1, :, :]
        # prediction = self.args.scaler.inv_transform(outputs)
        loss = self.loss(prediction, reals, 0.0)
        mape = utils.masked_mape(prediction, reals, 0.0).item()
        rmse = utils.masked_rmse(prediction, reals, 0.0).item()
        return loss.item(), mape, rmse
