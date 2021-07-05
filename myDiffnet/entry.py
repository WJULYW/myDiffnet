'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''

import sys, os, argparse

sys.path.append(os.path.join(os.getcwd(), 'class'))

from ParseConf import ParseConf
from DataUtil import DataUtil
from DataModule import DataModule


def executeTrainModel(config_path, model_name):
    print(config_path)
    conf = ParseConf(config_path)
    conf.parseConf()
    print(conf.topk)

    model=eval(model_name)
    model=model(conf)

    data=DataUtil(conf)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--gpu', nargs='?', help='available gpu id')

    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    device_id = args.gpu

    if device_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), 'conf/%s_%s.ini' % (data_name, model_name))
    executeTrainModel(config_path, model_name)
