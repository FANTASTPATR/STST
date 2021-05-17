import torch
import numpy as np
import argparse
import time

from torch.utils.tensorboard import SummaryWriter

import utils
from holder import Holder

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help='device name')
parser.add_argument('--data', type=str, default='PEMS-BAY', help='dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument('--epochs', type=int, default=500, help='training epoch')
parser.add_argument("--seq_in", type=int, default=12, help='historical length')
parser.add_argument("--seq_out", type=int, default=12, help='prediction length')
parser.add_argument("--seed", type=int, default=520, help='random seed')
parser.add_argument("--clip", type=float, default=5., help='gradient clip')
# ------lr=0.0001-----------------
parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
# ------dropout=0.2-----------------
parser.add_argument("--dropout", type=float, default=0.2, help='dropout rate')
# ------weight_decay=0.000001-----------------
parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay rate')
# -----snp_len=3------------------
parser.add_argument("--snp_len", type=int, default=4, help='the number of snp to form a group')
# -----num_heads=8------------------
parser.add_argument("--num_heads", type=int, default=8, help='heads (GAT),channel (GCN)')
# -----conv_layers=5------------------
parser.add_argument("--conv_layers", type=int, default=5, help='convolution layers')
# -----connect_type=adj------------------
parser.add_argument("--uni_direction", type=bool, default=True, help='message passing of uni-directional')
parser.add_argument("--connect_type", type=str, default="identity", help='connect with the adjacency matrix')
parser.add_argument("--dense_conn", type=bool, default=True, help='dense connections')
parser.add_argument("--recording", type=bool, default=True, help='whether recording')
parser.add_argument("--sync", type=bool, default=True, help='whether sync')
parser.add_argument("--meta_decode", type=bool, default=False, help='apply meta learning to decoding')
parser.add_argument("--comment", type=str, default="12-12,bay,snp4")

args = parser.parse_args()

if args.data == "METR-LA":
    args.data_file = "./data/METR-LA"
    args.adj_data = "./data/sensor_graph/adj_mx.pkl"
    args.num_node = 207
    args.in_dim = 2
    args.task = "speed"

elif args.data == "PEMS-BAY":
    args.data_file = "./data/PEMS-BAY"
    args.adj_data = "./data/sensor_graph/adj_mx_bay.pkl"
    args.num_node = 325
    args.in_dim = 2
    args.task = "speed"

elif args.data == "PEMS-D3":
    args.data_file = "./data/PEMS-D3"
    args.adj_data = "./data/sensor_graph/pems03.csv"
    args.num_node = 358
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D4":
    args.data_file = "./data/PEMS-D4"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D7":
    args.data_file = "./data/PEMS-D7"
    args.adj_data = "./data/sensor_graph/PEMS07.csv"
    args.num_node = 883
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D8":
    args.data_file = "./data/PEMS-D8"
    args.adj_data = "./data/sensor_graph/distance_pemsd8.csv"
    args.num_node = 170
    args.in_dim = 1
    args.task = "flow"



if args.recording:
    utils.record_info(str(args), "./records/" + args.comment)
    utils.record_info("12-12,bay,snp4", "./records/" + args.comment)
    sw = SummaryWriter(comment=args.comment)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.device = torch.device(args.device)


def main():
    if args.task == "speed":
        args.adj_mx = torch.Tensor(utils.load_pickle(args.adj_data)[-1])
    elif args.task == "flow":
        args.adj_mx, _ = torch.Tensor(utils.get_adjacency_matrix(args.adj_data, args.num_node))
    dataloader = utils.load_dataset(args.data_file, args.batch_size, args.batch_size, args.batch_size)
    args.scaler = dataloader['scaler']

    print(str(args))
    engine = Holder(args)
    print("start training...")

    his_loss = []
    val_time = []
    train_time = []

    for epoch_num in range(args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainX = torch.Tensor(x).to(args.device)
            trainy = torch.Tensor(y).to(args.device)
            if args.task == "speed":
                metrics = engine.train(trainX, trainy[:, :, :, 0:1])
            elif args.task == "flow":
                metrics = engine.train(trainX, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % 200 == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                if args.recording:
                    utils.record_info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),
                                      "./records/" + args.comment)
        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        print("eval...")
        s1 = time.time()
        # dataloader['test_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            valx = torch.Tensor(x).to(args.device)
            valy = torch.Tensor(y).to(args.device)
            if args.task == "speed":
                metrics = engine.eval(valx, valy[:, :, :, 0:1])
            elif args.task == "flow":
                metrics = engine.eval(valx, valy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch_num, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        if args.recording:
            sw.add_scalar('Loss/train', mtrain_loss, global_step=epoch_num)
            sw.add_scalar('Loss/valid', mvalid_loss, global_step=epoch_num)
            sw.add_scalar('MAPE/train', mtrain_mape, global_step=epoch_num)
            sw.add_scalar('MAPE/valid', mvalid_mape, global_step=epoch_num)
            sw.add_scalar('RMSE/train', mtrain_rmse, global_step=epoch_num)
            sw.add_scalar('RMSE/valid', mvalid_rmse, global_step=epoch_num)
        log = 'Epoch: {:03d}, Train Loss: {:.4f} ,Train MAPE: {:.4f}, Train RMSE: {:.4f}, ' \
              'Valid Loss: {:.4f},  Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                         mvalid_mape,
                         mvalid_rmse,
                         (t2 - t1)),
              flush=True)
        if args.recording:
            utils.record_info(
                log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                           mvalid_mape,
                           mvalid_rmse,
                           (t2 - t1)),
                "./records/" + args.comment)

        # torch.save(engine.model, "./model_d7.pkl")
    # torch.save(engine.model.state_dict(), './parameter_12.pkl')
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
