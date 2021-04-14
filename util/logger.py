# util/logger.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import os
import logging
import datetime
import pytz

# --------------------------------------------------------------------------------------------------------
# LogCollector
# https://realpython.com/python-logging/
# --------------------------------------------------------------------------------------------------------
class LogCollector():
    
    def __init__(self,
                 logger_name:str,
                 time_zone:str = 'Asia/Bangkok',
                ):

        self.__create_log_dir()
        
        self.__date_now = datetime.datetime.now(pytz.timezone(time_zone)).strftime('%Y-%m-%d')
        self.__path = f'log/{str(self.__date_now)}'
        self.__logger_name = logger_name
        self.__log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

        
        # DEBUG > INFO > WARNING > ERROR > CRITICAL
        environment = os.environ.get('HASHTAG_ANALYZER_RUNNING_ENV', 'dev')

        if environment == 'dev':
            self.__log_level_print = logging.DEBUG
            self.__log_level_file = logging.DEBUG 
        elif environment == 'uat':
            self.__log_level_print = logging.INFO
            self.__log_level_file = logging.ERROR 
        elif environment == 'prod':
            self.__log_level_print = logging.INFO
            self.__log_level_file = logging.ERROR 
        else:
            self.__log_level_print = logging.DEBUG
            self.__log_level_file = logging.DEBUG  

        # Create logger        
        self.logger = logging.getLogger(self.__logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create handlers
        print_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.__path)
        
        # Set Level of handlers
        print_handler.setLevel(self.__log_level_print)
        file_handler.setLevel(self.__log_level_file)
        
        # Create formatters 
        handler_format = logging.Formatter(self.__log_format)
        print_handler.setFormatter(handler_format)
        file_handler.setFormatter(handler_format)
        
        # Add handlers to the logger
        self.logger.addHandler(print_handler)
        self.logger.addHandler(file_handler)

    def __create_log_dir(self):

        try:
            os.mkdir('log/')
        except FileExistsError:
            pass

    def collect(self, level:str, message:object):
        ''' 
        Call the collect method everywhere you want to track back and keep in "log" directory, 
        the name of files depend on date in format 'yyyy-mm-dd'.
        
        Keyword arguments:
        level   -- can be "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL" ordered by the impact of criticalness increasing.
        message -- the strings to print out and keep in log file.
        
        ps.
        In development, logs will be printed and keep when the logs are greater or equal to the WARNING level.
        In production, logs will be only kept if the logs are greater than or equal to the ERROR level.
        '''
        level_dict = {  'debug':self.logger.debug, 
                        'info':self.logger.info, 
                        'warning':self.logger.warning, 
                        'error':self.logger.error, 
                        'critical':self.logger.critical}

        level = level.lower()
        printer = level_dict.get(level, self.logger.debug)
        
        if isinstance(message, str):
            printer(message)
        else:
            obj_str = message.__str__()
            message = '\n'+ obj_str
            printer(message)

    
if __name__ == '__main__':
    
    # Example of using
    env = 'dev'
    log_obj = LogCollector(logger_name='test')

    import pandas as pd
    data = pd.DataFrame(([1,2,3,4],[1,2,3,4]))


    log_obj.collect('DEBUG', data)
    log_obj.collect('INFO', data)
    log_obj.collect('WARNING', data)
    log_obj.collect('ERROR', data)
    log_obj.collect('CRITICAL', data)



# Ref.
