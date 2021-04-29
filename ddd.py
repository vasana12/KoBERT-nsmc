import inspect

def ss():
    this_function_name = inspect.currentframe().f_code.co_name
    print(this_function_name)
ss()