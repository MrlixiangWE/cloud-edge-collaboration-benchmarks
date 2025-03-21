import time

from CetStem.communication import start_zmq
from CetStem.utils.codepackage import codepackage

ti = time.time()
res = codepackage('./cetstem_detect.py', 'edge', demand='trest', reuse='run_infer', task='**')
print(time.time()-ti)
start_zmq(res)
print(time.time()-ti)
