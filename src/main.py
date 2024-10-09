import torch
import argparse
import os
from model import *
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def parse_args():
    parser = argparse.ArgumentParser( )
    parser.add_argument('--dataroot', type=str, default="../data/data.csv")
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--nhidden_encoder', type=int, default=64)
    parser.add_argument('--nhidden_decoder', type=int, default=64)
    parser.add_argument('--ntimestep', type=int, default=6)
    parser.add_argument('--predict_day', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    return args


def read_data(input_path, debug=True):
    df = pd.read_csv(input_path)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'TM' and x != 'LZ_Z']].to_numpy()
    y = np.array(df.LZ_Z)

    return X, y


def main():
    args = parse_args()
    X, y = read_data(args.dataroot, debug=False)
    model = EKLT(
        X,
        y,
        args.ntimestep,
        args.predict_day,
        args.nhidden_encoder,
        args.nhidden_decoder,
        args.batchsize,
        args.lr,
        args.epochs
    )
    print("==> Start training ...")
    model.train()
    y_pred = model.test()

if __name__ == '__main__':
    main()
