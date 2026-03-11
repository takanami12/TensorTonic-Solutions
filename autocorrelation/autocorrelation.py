def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    mean = ( 1 / len(series)) * sum(series)
    var = sum([(x - mean) ** 2 for x in series])
    res = [1] * (max_lag + 1)
    if var != 0:
        for i in range(1, max_lag+1):
            r = sum([(series[t] - mean) * (series[t+i] - mean) for t in range(len(series) - i)])
            res[i] = r / var
    else:
        res[1:] = [0] * max_lag
    return res