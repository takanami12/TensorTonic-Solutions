import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions)

    if predictions.shape[0] > 1:

        labels = set(predictions.flatten().tolist())
    
        votes = [{label: predictions.T[:][i].flatten().tolist().count(label) for label in labels} for i in range(predictions.shape[1])]
    
        result = [min(vote.items(), key=lambda x: (-x[1], x[0]))[0] for vote in votes]
        return result    

    else:
        return predictions.flatten().tolist()