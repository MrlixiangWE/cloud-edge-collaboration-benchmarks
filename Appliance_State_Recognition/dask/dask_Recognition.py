# -*- coding:utf-8 -*-

import numpy
import numpy as np
import torch
import socket
import time

from dask.distributed import Client

t_t = time.time()
dataSet = {}
devlists = ["Lamp1", "Lamp2", "Humidifier", "Television", "AirCleaner", "Roomba", "MicrowaveOven", "Refrigerator",
            "Monitor", "Touchpad1", "Charger", "Laptop", "Display", "Touchpad2", "PC"]
datapath = '../model/CurrentDSet/FinalCurrentData.npz'
labelpath = '../model/CurrentDSet/FinalLabel.npz'
modelpath = '../model/model_weights/checkpoint/model'
client = Client('192.168.3.22:8786')


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
        total_accuracy = total_num / len(a)
        # print(f'accuracy of {devlists[thisdevIndex]}:', total_accuracy)
        ac = ac + total_num
        s = s + len(a)
    return ac / s


# def get_host_ip():
#     hostname = socket.gethostname()
#     ip = socket.gethostbyname(hostname)
#     return ip
def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to actually connect
        s.connect(('180.76.76.76', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def task(x):
    t_1 = time.time()
    res = infer()
    return {'t_s': t_1, 'use_time': time.time() - t_1, 'ip': get_host_ip(), 'res': res}


futures = client.map(task, range(4))
results = client.gather(futures)
# 收集结果
results_ = []
for result in results:
    res_t = result.get('t_s') - t_t
    results_.append(
        {'ip': result.get('ip'), 'use_time': result.get('use_time'), 't_s': res_t, 'res': result.get('res')})
# 打印结果
print(results_)
print('all tasks done', time.time() - t_t)

# 关闭客户端连接
client.close()
