# -*- coding:utf-8 -*-

import numpy
import numpy as np
import torch
from CetStem.funcpriority import Stem
from CetStem.utils.thredNode import startRun
from collections import defaultdict

devlists = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven", "Refrigerator",
            "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]
# datapath = '/app/appexample/elect/CurrentDSet/FinalCurrentData.npz'
# labelpath = '/app/appexample/elect/CurrentDSet/FinalLabel.npz'
# modelpath = '/app/appexample/elect/model_weights/checkpoint/model'
datapath = '../../model/CurrentDSet/FinalCurrentData.npz'
labelpath = '../../model/CurrentDSet/FinalLabel.npz'
modelpath = '../../model/model_weights/checkpoint/model'

NODECLUSTER = defaultdict(list)
# NODECLUSTER['cloud'] = ['192.168.143.12']
NODECLUSTER['cloud'] = ['192.168.3.22']

NODECLUSTER['edge'] = ['192.168.3.26','192.168.3.27','192.168.3.28','192.168.3.37']
address_server = "tcp://"+NODECLUSTER.get('cloud')[0]+':5560'
address_client = "tcp://"+NODECLUSTER.get('cloud')[0]+':5559'
address_pull = "tcp://"+NODECLUSTER.get('cloud')[0]+':5557'


def infer():
    data = np.load(datapath, allow_pickle=True)
    labels = np.load(labelpath, allow_pickle=True)
    devlist_len = len(devlists)
    app = labels.files
    ac = 0
    s = 0
    DEVICE = torch.device('cpu')
    for thisdevIndex in range(devlist_len):
        name = app[thisdevIndex]

        target = labels[name]
        model_checkpoint_save_path = modelpath + str(thisdevIndex) + '.ckpt'
        model = torch.load(model_checkpoint_save_path)
        pred_result = model(torch.from_numpy(data['TotalCurrent'].astype(numpy.float32)))  # zhe
        pred_y = pred_result.argmax(axis=1).numpy()

        a = torch.tensor(pred_y).to(DEVICE)
        b = torch.tensor(target).to(DEVICE)
        total_num = torch.eq(a, b).sum().item()
        ac = ac + total_num
        s = s + len(a)
    return ac / s


@Stem('edge')
def run_infer():
    res = infer()
    return res


@Stem('cloud')
def mainCloud():
    startRun(address_pull, zmq_pull_timeout=1000)
    return 'cloud run over'
