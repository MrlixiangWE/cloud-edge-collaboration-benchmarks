import time

from CetStem.communication import recv, zmq_push, zmq_client
from CetStem.funcpriority import branch
import json

ti = time.time()
res = recv("tcp://192.168.3.22:4245")
print('接收时间:', time.time()-ti)
print(res)
t_ = time.time()
response = branch(json.loads(res), 'edge', 'run_infer')
print(response)
print('解析时间:', time.time()-t_)
zmq_push("tcp://192.168.3.22:5557", response)
print('解析时间1:', time.time()-t_)

