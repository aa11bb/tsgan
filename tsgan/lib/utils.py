import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def analyze_tensor_variables(variables=tf.trainable_variables(), print_info=False):
    """Prints the names and shapes of the variables.

    Args:
      variables: list of variables, for example tf.global_variables().
      print_info: Optional, if true print variables and their shape.

    Returns:
      (total size of the variables, total bytes of the variables)

    Note:
        refer from tensorflow.contrib.slim.model_analyzer.analyze_vars
    """
    if print_info:
        print('---------')
        print('Variables: name (type shape) [size]')
        print('---------')
    total_size = 0
    total_bytes = 0
    var_str_list = list()
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes
        var_str = "({}, {}, [var_size:{}, bytes: {}])". \
            format(var.name, slim.model_analyzer.tensor_description(var),
                   var_size, var_bytes)
        var_str_list.append(var_str)
        if print_info:
            print(var_str)
    total_size_str = 'Total size of variables: %d' % total_size
    total_bytes_str = 'Total bytes of variables: %d' % total_bytes
    var_str_list.append(total_size_str)
    var_str_list.append(total_bytes_str)
    if print_info:
        print(total_size_str)
        print(total_bytes_str)

    return var_str_list


def analyze_object_variables(oj, print_info=False):
    var_str_list = []
    for key in oj.__dict__:
        str_var = "{key}='{value}'".format(key=key,
                                           value=oj.__dict__[key])
        var_str_list.append(str_var)
        if print_info:
            print(str_var)
    return var_str_list


def save_variables_to_file(file_name, variables):
    with open(file_name, "w") as file:
        for var in variables:
            file.write(str(var) + "\n")


