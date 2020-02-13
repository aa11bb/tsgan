import matplotlib.pyplot as plt


def plot_series(samples, out_path):
    f, axes = plt.subplots(samples.shape[0] // 2, 2)
    axes = axes.flat[:]
    #title = out_path.split('/')[-1].split('.')[0]
    #plt.suptitle(title)
    for i, ax in enumerate(axes):
        ax.plot(samples[i])

    plt.savefig(out_path)
    plt.close(f)

def plot_gan_loss(g_losses, d_losses, save_path):
    fig = plt.figure()
    plt.title('loss')
    plt.plot(g_losses, label='g_loss')
    plt.plot(d_losses, label='d_loss')
    plt.xlabel('index')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.close(fig)


def plot_dic(dic_values, title=None, xlabel=None, ylabel=None, ylim=None, save_path=None):
    fig = plt.figure()
    for key, y in dic_values.items():
        plt.plot(y, label=key)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
