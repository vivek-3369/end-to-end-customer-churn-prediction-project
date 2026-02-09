import sys 
from logger import logging


def error_message(error, error_detail:sys) :
    '''
    Creating a custom detailed error message 
    '''

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_detail_message = f"Error occured in python script [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_detail_message


class CustomException(Exception) :
    '''
    Creating a Customer Error Handling System
    '''

    def __init__(self, error, error_details:sys) :
        super().__init__(error)
        self.error = error_message(error, error_details) 

    def __str__(self) :
        return self.error