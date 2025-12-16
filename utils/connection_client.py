# --------------------------------------------------------
# ZTE_project
# Transport video and frame to Server
# --------------------------------------------------------
import datetime
from socket import *
import os
import time
from utils.config import get_config
from utils.logger import create_logger

this_dir = os.path.split(os.path.realpath(__file__))[0]
now_date = datetime.datetime.now().strftime('%Y-%m-%d')
main_opt = get_config()
video_dir = main_opt.video_path
img_dir = main_opt.frame_path
logger = create_logger(output_dir=this_dir.replace('/utils', '') + '/log/transport', mode=now_date, name='transport')
HOST = main_opt.ip  # or 'localhost'
PORT = main_opt.transport_port
BUFF = 1024
ADDR = (HOST, PORT)
tcpCliSock = socket(AF_INET, SOCK_STREAM)
all_size = 0


def client(path, name, trans_type):
    global all_size
    f = open(path, 'rb')
    file_size = os.stat(path).st_size
    all_size += file_size
    tcpCliSock.sendall(bytes(str(file_size) + '-' + name + '-' + trans_type, 'utf-8'))  # size_name
    if str(tcpCliSock.recv(BUFF), 'utf-8') == 'info yes':
        has_sent = 0
        while has_sent != file_size:
            data = f.read(1024)
            tcpCliSock.sendall(data)
            has_sent += len(data)
        f.close()
        if str(tcpCliSock.recv(BUFF), 'utf-8') == 'file yes':
            return True
        else:
            return False


def video_transport(video_name):
    # 视频
    for i in video_name:
        path = video_dir + '/' + i
        if client(path, i, 'video'):
            logger.info('send successfully---' + path)
            continue
        else:
            break


def frame_transport(img_name):
    # 分帧图片
    for i in img_name:
        path = img_dir + '/' + i
        if client(path, i, 'frame'):
            logger.info('send successfully---' + path)
            continue
        else:
            break


def over():
    logger.info('send over!')
    tcpCliSock.sendall(bytes('send over', 'utf-8'))
    tcpCliSock.close()


def transport():
    global all_size
    start_time = time.time()
    do_connect = True
    video_name = os.listdir(video_dir)
    img_name = os.listdir(img_dir)
    try:
        tcpCliSock.connect((HOST, PORT))
        logger.info("Connected to %s on port %s" % (HOST, PORT))
        logger.info('Transporting formally')
    except error as e:
        logger.error("Connected to %s on %s port failed: %s" % (HOST, PORT, e))
        logger.error('Please check the server and retry')
        do_connect = False
    if do_connect:
        video_transport(video_name)
        frame_transport(img_name)
        over()
        end_time = time.time()
        spend_time = end_time - start_time
        return [spend_time, all_size]
    else:
        return False


if __name__ == '__main__':
    transport()
