# init_sys.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import os
import sys
import warnings

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))

from config_loader import ConfigLoader

# --------------------------------------------------------------------------------------------------------
# InitSystem
# --------------------------------------------------------------------------------------------------------

class InitSystem:
    def __init__(self, running_env='dev'):

        self.__running_env = running_env

        # Set OS Envirionment Variables
        os.environ["HASHTAG_ANALYZER_RUNNING_ENV"] =  self.__running_env

        # Load Config
        self.__cl = ConfigLoader()

    # Warning for Development Environment
    def __check_dev(class_method):
        def method_wrapper(self, *arg, **kwarg):
            if self.__running_env == 'dev':
                warnings.warn('You can use this method only in development environment only.')
                return class_method(self, *arg, **kwarg)
            else:
                raise Exception('You can use this method only in development environment only.')

        return method_wrapper

    def init(self):
        # Enable/Disable GPU
        use_gpu = self.__cl.get('GPU', 'use_gpu', data_type=bool)

        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

        return self.__cl
    


    @__check_dev
    def get_config_loader(self):
        return self.__cl


            
if __name__ == '__main__':
    InitSystem()
    pass