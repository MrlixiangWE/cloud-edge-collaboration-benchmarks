import json
import os
import time
from collections import namedtuple
import _pickle as cPickle
# from functools import lru_cache
from inspect import isgenerator

from flask import Flask, request, Response
from flask_cors import CORS
from cachelib import SimpleCache

cache = SimpleCache()
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
app.config["JSON_AS_ASCII"] = False
TaskResult = namedtuple('TaskResult', ['code', 'result', 'task', 'node'])


@app.route('/manage/<path:path>')
# @lru_cache()
def manage(path):
    if path:
        recv_data = request.get_data()
        if recv_data:
            recv_data = json.loads(recv_data)
            code = recv_data.get('code')
            task = recv_data.get('task')
            node = recv_data.get('node')
            taskres = TaskResult(code=code, result=None, task=task, node=node)
            cache.set('tasktuple', taskres, timeout=300)
            print('code deploy...')
            return {'code': 200, 'msg': '代码部署完成'}


@app.route('/resource/<path:path>')
# @lru_cache()
def resourse(path):
    ti = time.time()
    if path:
        taskres = cache.get('tasktuple')
        code = taskres.code
        node = taskres.node
        temp = {}
        fun = compile(code, '<string>', 'exec')
        exec(fun, temp)
        ret = temp['main'+node.capitalize()]()
        print('code run...')
        print('ret', ret)
        if isgenerator(ret):
            next(ret)
            print(time.time() - ti)
            return Response(ret, mimetype='multipart/x-mixed-replace; boundary=frame')
        taskres = taskres._replace(result=ret)
        print('执行时间',time.time() - ti)
        # return taskres._asdict()
        return str(time.time() - ti)


@app.route('/filedown', methods=['POST'])
def filedown():
    datafile = request.files['file']
    filename = datafile.filename
    base_path = os.path.dirname(__file__)
    upload_path = os.path.join(base_path, filename)
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    try:
        datafile.save(os.path.join(upload_path, filename))
        return {'code': 200, 'msg': '数据{}上传成功'.format(filename)}
    except Exception as e:
        return {'code': 100981, 'msg': '数据上传失败'.format(filename), 'error': e}


@app.route('/data/<name>')
def reqdata(name):
    if name == 'detect_device':
        return
    if name == 'object_detect':
        return 'http://172.16.101.57:5006/static/vtest.avi'
    if name == 'object_track':
        return 'http://172.16.101.57:5006/static/p.mp4'


if __name__ == '__main__':
    app.run('0.0.0.0', port=5006, debug=False)
