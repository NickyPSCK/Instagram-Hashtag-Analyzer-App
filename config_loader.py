# config_loader.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import os
import base64
import configparser
# --------------------------------------------------------------------------------------------------------
# ConfigLoader
# --------------------------------------------------------------------------------------------------------
class ConfigLoader:
    def __init__(self):
        
        self.__env = os.environ.get('RUNNING_ENV', None)

        self.__default_dir = os.path.dirname(os.path.realpath(__file__))

        if self.__env == 'uat':
            self.__arg_config_dir = os.path.join(self.__default_dir, *'config/config_uat.ini'.split('/'))
        elif self.__env == 'prod':
            self.__arg_config_dir = os.path.join(self.__default_dir, *'config/config_prod.ini'.split('/'))
        else:
            self.__arg_config_dir = os.path.join(self.__default_dir, *'config/config_dev.ini'.split('/'))

        self.__config = configparser.ConfigParser()
        self.__config.read(self.__arg_config_dir)

        # Decode Password
        self.__config['login']['password'] = self.__decode_base_64(self.__config['login']['password'])

    def __encode_base_64(self, message):
        message_bytes = message.encode('ascii')
        base64_bytes = base64.b64encode(message_bytes)
        base64_message = base64_bytes.decode('ascii')
        return base64_message

    def __decode_base_64(slef, base64_message):
        base64_bytes = base64_message.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)
        message = message_bytes.decode('ascii')
        return message

    def save_new_password(self, password, env='dev'):
        if env == 'uat':
            config_dir = os.path.join(self.__default_dir, *'config/config_uat.ini'.split('/'))
        elif env == 'prod':
            config_dir = os.path.join(self.__default_dir, *'config/config_prod.ini'.split('/'))
        else:
            config_dir = os.path.join(self.__default_dir, *'config/config_dev.ini'.split('/'))

        self.__config['login']['password'] = self.__encode_base_64(password)

        with open(config_dir, 'w') as configfile:
            self.__config.write(configfile)

        self.__config['login']['password'] = password


    def get(self, section, key, data_type=str):
        if data_type is bool:
            value = self.__config.getboolean(section, key, fallback=None)
        else:
            value = data_type(self.__config.get(section, key, fallback=None))
        if value is not None:
            return value
        else:
            raise Exception('Invalid section or key.')


if __name__ == '__main__':

    # # Change Password
    # cl = ConfigLoader()
    # cl.save_new_password('type your new password here')
    pass



