import torch
import argparse
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:2", help='device name')
parser.add_argument('--data', type=str, default='METR-LA', help='dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument("--seq_out", type=int, default=12, help='prediction length')
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

elif args.data == "PEMS-D4":
    args.data_file = "./data/PEMS-D4"
    args.adj_data = "./data/sensor_graph/distance_pemsd4.csv"
    args.num_node = 307
    args.in_dim = 1
    args.task = "flow"

elif args.data == "PEMS-D8":
    args.data_file = "./data/PEMS-D8"
    args.adj_data = "./data/sensor_graph/distance_pemsd8.csv"
    args.num_node = 170
    args.in_dim = 1
    args.task = "flow"

dataloader = utils.load_dataset(args.data_file, args.batch_size, args.batch_size, args.batch_size)
args.scaler = dataloader['scaler']
args.device = torch.device(args.device)


def test():
    model = torch.load("./model_metr_12.pkl", map_location=args.device)
    model.eval()
    test_mae = []
    test_mape = []
    test_rmse = []

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testX = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)
        if args.task == "speed":
            with torch.no_grad():
                outputs = model(testX)
            testy = testy[:, :3, :, 0:1][:, -1, :, :]
            prediction = args.scaler.inv_transform(outputs)[:, :3, :, :][:, -1, :, :]
            mae = utils.masked_mae(prediction, testy, 0.0).item()
            rmse = utils.masked_rmse(prediction, testy, 0.0).item()
            mape = utils.masked_mape(prediction, testy, 0.0).item()
            test_mae.append(mae)
            test_mape.append(mape)
            test_rmse.append(rmse)
    mae = np.mean(test_mae)
    mape = np.mean(test_mape)
    rmse = np.mean(test_rmse)
    log = 'Test Loss: {:.4f},  Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(mae, mape, rmse))


if __name__ == '__main__':
    test()
