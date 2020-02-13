import numpy as np
import sklearn.preprocessing as skpre

#########################
# Normalize time series
TS_NORMALIZE_TYPES = ['znorm', 'znorm-center', 'scale-minmax', 'scale-maxabs',
                      'nonlinear-quantile','norm-l1', 'norm-l2', 'norm-max',
                      'sigmoid', 'tanh']
def normalize(X, mode, **kwargs):
    """
    NOTE: The axis must be set to 1 for time series so as to standardize each sample that is 
    different from the common spreadsheet data and image data. Therefore, it is dose not has the 
    training process to memorize the statistics calculated on training set. In addition, the 
    Transformer classes realized in sklearn can not be used for time series because they are 
    realized to independently normalize/scale/standardize each feature otherwise each sample.
    :param X: 
    :param mode: 
    :param kwargs: 
    :return: 
    """
    axis = 1
    ret = None
    ## Standardization, or mean removal and variance scaling:
    # they might behave badly if the individual features do not more or less look like standard
    # normally distributed data.
    if mode == 'znorm':
        ret = skpre.scale(X, axis=axis, **kwargs)
    elif mode == 'znorm-center':
        ret = skpre.scale(X, axis=axis, with_std=False, **kwargs)
    # MinMax, MaxAbs:
    # The motivation to use this scaling include robustness to very small standard deviations
    # of features and preserving zero entries in sparse data.
    elif mode == 'scale-minmax': # [min, max], default is [0, 1]
        ret = skpre.minmax_scale(X, axis=axis, **kwargs)
    elif mode == 'scale-maxabs': # [-1, 1]
        ret = skpre.maxabs_scale(X, axis=axis, **kwargs)
    # Scaling data with outliers:
    elif mode == 'scale-robust':
        ret = skpre.robust_scale(X, axis=axis, **kwargs)
    ## Non-linear transformation
    # Quantile transforms maps data to a uniform distribution. It should be noted that such
    # operation distort correlations and distance within and across features.
    elif mode == 'nonlinear-quantile':
        ret = skpre.quantile_transform(X, axis=axis, **kwargs)
    # Power transforms aim o map data from any distribution to as close to a Gaussian
    # distribution as possible in order to stabilize variance and minimize skewness.
    elif mode == 'nonlinear-power': # TODO: this function can not be found.
        raise ValueError("This option did not be realized!")
    ## Normalization
    # Normalization is the process of scaling individual samples to have unit norm.
    elif mode == 'norm-l1':
        ret = skpre.normalize(X, norm='l1', axis=axis, **kwargs)
    elif mode == 'norm-l2':
        ret = skpre.normalize(X, norm='l2', axis=axis, **kwargs)
    elif mode == 'norm-max':
        ret = skpre.normalize(X, norm='max', axis=axis, **kwargs)
     ## some other often-used non-linear normalization methods
    elif mode == 'sigmoid':
        sigmoid = (lambda x: 1.0 / (1.0 + np.exp(-x)))
        ret = sigmoid(X)
    elif mode == 'tanh':
        ret = np.tanh(X, **kwargs)
    else:
        raise ValueError("No normalization type with {} found !".format(mode))

    return ret


#########################
# Reference: MCNN in https://github.com/hfawaz/dl-4-tsc

def slice(x, y, l_sliced):
    n = x.shape[0]
    l = x.shape[1]
    dim = x.shape[2]  # for MTS

    l_sliced = int(l * l_sliced) if l_sliced<1.0 else int(l_sliced)
    # l_sliced = int(l * ratio)
    n_inc = l - l_sliced + 1  # one ori derives n_inc instances.
    n_sliced = n * n_inc

    x_new = np.zeros((n_sliced, l_sliced, dim))
    y_new = np.zeros((n_sliced, )+y.shape[1:])
    for i in range(n):
        for j in range(n_inc):
            x_new[i * n_inc + j, :, :] = x[i, j:(j + l_sliced), :]
            y_new[i * n_inc + j] = y[i]
    return x_new, y_new, n_inc


def down_sample(x, span, offset=0):
    n = x.shape[0]
    l = x.shape[1]
    dim = x.shape[2] # for MTS
    last_one = 0
    if l % span > offset: # the tail can be included.
        last_one = 1
    l_new = int(np.floor(l / span)) + last_one
    out = np.zeros((n, l_new, dim))
    for i in range(l_new):
        out[:, i] = x[:, offset + span * i]

    return out

def down_sample_batch(x, span_base, step_size, n_step):
    if x.shape[1] == 26: # the case for dataset JapaneseVowels MTS
        return (None, []) # too short to apply down sampling
    if n_step == 0:
        return (None, [])

    out_ts = down_sample(x, span_base, 0)
    out_lengths = [out_ts.shape[1]]

    for i in range(1, n_step):
        span = span_base + step_size * i
        if span > x.shape[1]: # exceeded the length of ts
            break
        for offset in range(0, 1):
            series_new = down_sample(x, span, offset)
            out_lengths.append(series_new.shape[1])
            out_ts = np.concatenate([out_ts, series_new], axis=1)

    return (out_ts, out_lengths)


def moving_avg(x, win_size):
    n = x.shape[0]
    l = x.shape[1]
    dim = x.shape[2] # for MTS
    out_l = l - win_size + 1
    out_ts = np.zeros((n, out_l, dim))
    for i in range(out_l):
        out_ts[:, i] = np.mean(x[:, i:(i+win_size)], axis=1)
    return out_ts

def moving_avg_batch(x, win_base, step_size, n_steps):
    if n_steps == 0:
        return (None, [])
    out_ts = moving_avg(x, win_base)
    out_lengths = [out_ts.shape[1]]
    for i in range(1, n_steps):
        win_size = win_base + step_size * i
        if win_size > x.shape[1]:
            continue
        series_new = moving_avg(x, win_size)
        out_lengths.append(series_new.shape[1])
        out_ts = np.concatenate([out_ts, series_new], axis=1)

    return (out_ts, out_lengths)


#########################
# Reference: t-leNet in https://github.com/hfawaz/dl-4-tsc

def warping(ts, ratio):
    n = ts.shape[0]
    l = ts.shape[1]
    dim = ts.shape[2]  # for MTS

    t = np.arange(0, l, ratio)
    tp = np.arange(0, l)
    l_new = len(t)
    ts_new = np.zeros((n, l_new, dim), dtype=np.float64)
    for i in range(n):
        for j in range(dim):
            ts_new[i, :, j] = np.interp(t, tp, ts[i, :, j])

    return ts_new