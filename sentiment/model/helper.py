"""
Author: Lu ZhiPing
email: lu.zhiping@u.nus.edu
"""
import json
import numpy as np


# NumPy dtype and Python Type converter
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
