import os
import time
import threading
import PackageMessage_pb2

try:
    import zmq
except ImportError:
    print("Please install zmq:\n\n    python -m pip install zmq\n    python -m pip install protobuf==3.20.3\n")
    exit(0)

## 此文件用于模拟调试 zmq 消息的收发

# 1. python -m pip install protobuf==3.20.3
# 2. python -m pip install zmq
# 3. 启动消息中心, 配置收发均为本机IP (必须启动消息中心才能收发消息) (消息监控程序可以监控消息)

# 4. 连接消息中心设置的发布地址, socketPub 用于发消息
contextPub = zmq.Context()
socketPub = contextPub.socket(zmq.PUB)
socketPub.connect("tcp://192.168.0.132:1234")  # 消息中心设置的发布地址

# 5. 连接消息中心设置的订阅地址, socketSub 用于接收消息
contextSub = zmq.Context()
socketSub = contextSub.socket(zmq.SUB)
socketSub.connect("tcp://192.168.0.132:5678")  # 消息中心设置的订阅地址

# 6.设置订阅的 topic. 只有订阅了 topic 才能收到消息, 不要订阅自己发送消息所用的 topic, 否则会收到自己的消息
socketSub.setsockopt_string(zmq.SUBSCRIBE, 'Rb_ObjectDetection')


def zmqPub():
    pm = PackageMessage_pb2.PackageMessage()
    # 发送PackageMessage
    pm.SessionId = '101'
    pm.Token = '1'
    pm.robotID = '8'
    pm.To = 'Rb_ObjectDetection'
    pm.From = 'Rb_ObjectDetection'
    pm.ToPic = 'Rb_ObjectDetection'
    pm.CallMessage.Function = "Leak"
    pm.ResultMessage.ErrorCode = 0

    pm.CallMessage.Parameters.append('D:\\User\\Desktop\\task1'.encode())
    pm.CallMessage.Parameters.append('no json file.'.encode())
    msg1 = pm.SerializeToString()
    pm.CallMessage.Parameters[:] = []

    pm.CallMessage.Parameters.append('D:\\User\\Desktop\\task2'.encode())
    pm.CallMessage.Parameters.append('no json file.'.encode())
    msg2 = pm.SerializeToString()
    pm.CallMessage.Parameters[:] = []

    pm.CallMessage.Parameters.append('D:\\User\\Desktop\\task3'.encode())
    pm.CallMessage.Parameters.append('no json file.'.encode())
    msg3 = pm.SerializeToString()
    pm.CallMessage.Parameters[:] = []

    pm.CallMessage.Parameters.append('D:\\User\\Desktop\\task4'.encode())
    pm.CallMessage.Parameters.append('no json file.'.encode())
    msg4 = pm.SerializeToString()
    pm.CallMessage.Parameters[:] = []

    for _ in range(10):
        socketPub.send("Rb_ObjectDetection".encode("utf8"), zmq.SNDMORE)
        socketPub.send(" ".encode("utf8"), zmq.SNDMORE)
        socketPub.send(msg1)
        print("send file complete 1")
        # time.sleep(1)

    for _ in range(10):
        socketPub.send("Rb_ObjectDetection".encode("utf8"), zmq.SNDMORE)
        socketPub.send(" ".encode("utf8"), zmq.SNDMORE)
        socketPub.send(msg2)
        print("send file complete 2")
        # time.sleep(1)

    for _ in range(10):
        socketPub.send("Rb_ObjectDetection".encode("utf8"), zmq.SNDMORE)
        socketPub.send(" ".encode("utf8"), zmq.SNDMORE)
        socketPub.send(msg3)
        print("send file complete 3")
        # time.sleep(1)

    for _ in range(10):
        socketPub.send("Rb_ObjectDetection".encode("utf8"), zmq.SNDMORE)
        socketPub.send(" ".encode("utf8"), zmq.SNDMORE)
        socketPub.send(msg4)
        print("send file complete 4")
        # time.sleep(1)


def zmqSub():
    print("======zmqSub======")
    pm_sub = PackageMessage_pb2.PackageMessage()

    while True:
        time.sleep(2)
        message = socketSub.recv()
        print(message)


if __name__ == '__main__':
    t1 = threading.Thread(target=zmqPub)
    t2 = threading.Thread(target=zmqSub)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
