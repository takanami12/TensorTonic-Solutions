import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    if len(points.shape) == 1:
        points.reshape(3)
        

        points = np.hstack([points, np.ones((1))])
        
        h = np.dot(T, points.T)
        
        return h[:-1]
    else:
        points.reshape(-1, 3)

        points = np.hstack([points, np.ones((points.shape[0], 1))])

        h = np.dot(T, points.T)
        
        h = h[:-1, :].T
        
        return h
    

    