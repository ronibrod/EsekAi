import numpy as np
from sklearn.preprocessing import FunctionTransformer
from .const import output_scaler_max_range

def scale_down(values):
    return np.array(values) / output_scaler_max_range
def scale_up(values):
    return np.array(values) * output_scaler_max_range
def get_output_scaler():
    return FunctionTransformer(scale_down, inverse_func=scale_up)
