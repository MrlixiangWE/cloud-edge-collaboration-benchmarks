import time

from CetStem.communication import recv, zmq_push
from CetStem.funcpriority import branch
import json

ti = time.time()
res = recv("tcp://192.168.3.22:4245")
print('接收时间:', time.time()-ti)
print(res)
t_ = time.time()
response = branch(json.loads(res), 'cloud', 'mainCloud')
# zmq_push("tcp://172.17.0.12:5557", response)
print('解析时间:', time.time()-t_)
print(response)
