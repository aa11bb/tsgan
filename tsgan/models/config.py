import os
import shutil


class Config(object):
    def __init__(self, X, data_name, out_fname, dim_h, dim_w, dim_c, state='train'):
        # data parameter
        self.X = X
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_c = dim_c
        self.dim_z = int(dim_h * 0.7)
        self.random_seed = 42

        # model parameter
        self.gf_dim = 64  # dimension of G filters in first conv layer
        self.df_dim = 64  # dimension of D filters in first conv layer

        # training parameter
        self.batch_size = min(16, len(self.X))
        self.g_lr = 0.0002
        self.d_lr = 0.0002
        self.g_beta1 = 0.5
        self.d_beta1 = 0.5
        self.n_epochs = 300
        self.acc_threshold = 0.75

        # log parameter
        self.data_name = data_name
        self.file_name = out_fname
        self.model_name = "{}_{}".format(self.file_name, self.data_name)
        self.freq_print = 1
        self.freq_plot = 20
        self.freq_log = 20
        self.n_samples = 10000

        self.dir_root = 'cache/{}'.format(self.file_name)
        self.dir_root_dataset = '{}/{}'.format(self.dir_root, self.data_name)
        self.dir_logs = os.path.join(self.dir_root_dataset, 'logs')
        self.dir_samples = os.path.join(self.dir_root_dataset, 'samples')
        self.dir_checkpoint = os.path.join(self.dir_root_dataset, 'checkpoint')

        # construct log file
        if state == 'train':
            if os.path.exists(self.dir_root_dataset):
                shutil.rmtree(self.dir_root_dataset)
            os.makedirs(self.dir_root_dataset)
            os.makedirs(self.dir_logs)
            os.makedirs(self.dir_samples)
            os.makedirs(self.dir_checkpoint)
