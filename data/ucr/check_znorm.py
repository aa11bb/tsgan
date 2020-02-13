from data import ucr

import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    dir_data = '../../dataset/UCR_TS_Archive_2015'
    dir_out = 'result'

    res = {'dataset':[], 'mean':[], 'std':[]}
    data_name_list = ucr.get_data_name_list(dir_data)
    for data_name in data_name_list:
        x_tr, y_tr, x_te, y_te, n_classes = ucr.load_ucr_flat(data_name, dir_data)
        x = np.vstack([x_tr, x_te])
        mean = np.round(np.mean(np.mean(x, axis=1)),2)
        std = np.round(np.mean(np.std(x, axis=1)),2)
        res['dataset'].append(data_name)
        res['mean'].append(mean)
        res['std'].append(std)

    df = pd.DataFrame(res)
    df.to_csv(os.path.join(dir_out, 'check_znorm.csv'), index=False)

