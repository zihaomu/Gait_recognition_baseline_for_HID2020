from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .model import SilhouetteDeep, SilhouetteNormal, YouOwnModel

def get_SilhouetteNormal(frame_num, pid_num):
    model = SilhouetteNormal(num_classes=pid_num)
    return model

def get_SilhouetteDeep(frame_num, pid_num):
    model = SilhouetteDeep(num_classes=pid_num)
    return model

def get_YouOwnModel(frame_num, pid_num):
    model = YouOwnModel(num_classes=pid_num)
    return model

def get_model(config):
    model_name = config.model.name
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    model = f(config.data.frame_num, config.data.pid_num)
    return model

def get_model_test(model_name):
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    return f()
    
if __name__ == '__main__':
    model_name = "SilhouetteNormal"
    f = get_model_test(model_name)
    print(f)
