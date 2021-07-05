'''
    author: Jiyao WANG
    e-mail: jiyaowang130@gmail.com
    release date:
'''

import configparser as cp
import re, os


class ParseConf:

    def __init__(self, config_path):
        self.config_path = config_path

    def processValue(self, section):
        for key, value in section:
            print(key, value)
            value = value.split(' ')
            dtype = value[0]
            value = value[1:]
            if value != None:
                if dtype == 'string':
                    self.conf_dict[key] = vars(self)[key] = value[0]
                elif dtype == 'int':
                    self.conf_dict[key] = vars(self)[key] = int(value[0])
                elif dtype == 'float':
                    self.conf_dict[key] = vars(self)[key] = float(value[0])
                elif dtype == 'list':
                    self.conf_dict[key] = vars(self)[key] = [i for i in value]
                elif dtype == 'int_list':
                    self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
                elif dtype == 'float_list':
                    self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
            else:
                print('%s value is None' % key)

    def parseConf(self):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf

        self.conf_dict = {}
        for section in conf.sections():
            self.processValue(conf.items(section))

        self.data_dir=os.path.join(os.getcwd(),'data/%s'%self.data_name)
        self.links_filename = os.path.join(os.getcwd(), 'data/%s/%s.links' % (self.data_name, self.data_name))
        self.user_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/user_vector.npy' % self.data_name)
        self.item_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/item_vector.npy' % self.data_name)
        self.pre_model = os.path.join(os.getcwd(), 'pretrain/%s/%s' % (self.data_name, self.pre_model))
