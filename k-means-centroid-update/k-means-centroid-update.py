def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    centroids = [[0] * len(points[0])] * k
    count = {a: assignments.count(a) for a in assignments}
    
    for i in range(len(points)):
        centroids[assignments[i]] = [sum(x) for x in zip(centroids[assignments[i]], points[i])]

    for key in count.keys():
        centroids[key] = [x / count[key] for x in centroids[key]]

    return centroids