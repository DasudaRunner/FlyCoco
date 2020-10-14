# coding:utf-8
# datetime:2020/9/24 12:33 上午
# author: haibo
import yaml
import numpy as np

def visMasks(img,mask,bbox=None,dump=False):
    pass

def readConfig(file_name):
    with open(file_name,'r') as f:
        data = yaml.load(f)
    return data

def print_dict(val,with_type=False):
    str_val = '{\n'
    for k,v in val.items():
        str_val+='   '
        str_val += '\'{}\': '.format(k)
        if isinstance(v,str):
            str_val += '\'{}\',\n'.format(v)
        else:
            str_val += '{},\n'.format(v)

    str_val = str_val[:-2]+str_val[-1]
    str_val+='}'
    print(str_val)

    if with_type:
        str_val = '{\n'
        for k,v in val.items():
            str_val+='   '
            str_val += '{}: <{}>\n'.format(k,type(v))