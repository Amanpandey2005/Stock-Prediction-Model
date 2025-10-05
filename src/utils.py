# Currently optional, can add helper functions here later
def rmse(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((y_true - y_pred)**2))
