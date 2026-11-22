import numpy as np
def precision(y_true, y_pred):
    # Your code here
    den=np.sum(y_pred)
    if den==0 :
        return 0.0
    num=np.dot(y_pred,y_true)
    return num/den
    
