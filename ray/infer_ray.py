# -*- coding:utf-8 -*-
import os

import numpy
import numpy as np
import torch
import ray
import time

dataSet = {}
devlists = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven", "Refrigerator",
            "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]
datapath = '../model/CurrentDSet/FinalCurrentData.npz'
labelpath = '../model/CurrentDSet/FinalLabel.npz'
modelpath = '../model/model_weights/checkpoint/model'



def infer():
    rootpath = os.getcwd()
    print("path check:"+rootpath+"\n")
#    pdata = os.path.join(rootpath,datapath)
#    plabel = os.path.join(rootpath,labelpath)
#    pmodel = os.path.join(rootpath,modelpath)
#    print(plabel)
#    print(pmodel+"\n")

    data = np.load(datapath, allow_pickle=True)
    labels = np.load(labelpath, allow_pickle=True)
    devlist_len = len(devlists)
    app = labels.files
    ac = 0
    s = 0
    DEVICE = torch.device('cpu')
    start_time = time.time()
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
        total_accuracy = total_num / len(a)
        # print(f'accuracy of {devlists[thisdevIndex]}:', total_accuracy)
        ac = ac + total_num
        s = s + len(a)
    return 'total accuracy:', ac / s

    # print('total time:', time() - start_time)
    # print('total accuracy:', ac / s)

@ray.remote(num_cpus=1)
def main():
    ti = time.time()
    res = infer()
    print(res)
    return ray._private.services.get_node_ip_address() + ":" + str(time.time() - ti) + ":" + str(ti)


if __name__ == "__main__":

    ray.init(address='auto')
    t1 = time.time()
    ips = ray.get([main.remote() for _ in range(4)])
    result = []
    for res in ips:
        node_ip, run_time, res_time = res.split(":")
        result.append({"node": node_ip, "runtime": run_time, "restime": float(res_time) - t1})
    print(result)
    print("all time: ", time.time() - t1)
    print("result: ", result)
