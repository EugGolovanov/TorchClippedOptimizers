import matplotlib.pyplot as plt


class PlotMaker:
    """
    A class for drawing graphs of loss functions and model metrics
    >>> plot_maker = PlotMaker("Test title", ["first optimizer name", "second_optimizer name", "third optimizer namae"], \
                     metric_name="IoU", loss_name="focal_loss")
    >>> history_example = [[1, 2, 3],       # for first optimizers
    >>>                    [1, 2, 3],       # for second optimizer
    >>>                    [1, 2, 3]]       # for third optimizer
    >>> plot_maker.draw_plot(history_one, history_two, history_three)
    """

    font_dict = {'fontsize': 14, 'fontweight': 'medium'}
    def __init__(self, main_title, names_optimizers, metric_name="accuracy", loss_name=""):
        self.main_title = main_title
        self.names_optimizers = names_optimizers
        self.metric_name = metric_name
        self.loss_name = loss_name

    def draw_plot(self, train_losses, train_acc, val_acc):
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(self.main_title, fontsize=20)

        ax1 = plt.subplot2grid((2, 5), (0, 0), rowspan=2, colspan=3)
        ax2 = plt.subplot2grid((2, 5), (0, 3), colspan=2)
        ax3 = plt.subplot2grid((2, 5), (1, 3), colspan=2)

        ax1.set_title(f"Train loss {self.loss_name}", fontdict=self.font_dict)
        ax2.set_title(f"Train {self.metric_name}", fontdict=self.font_dict)
        ax3.set_title(f"Valid {self.metric_name}", fontdict=self.font_dict)

        axes = [ax1, ax2, ax3]
        all_histories = [train_losses, train_acc, val_acc]

        for ax, loss_histories in zip(axes, all_histories):
            for ind, loss_history in enumerate(loss_histories):
                ax.plot(loss_history, label=self.names_optimizers[ind], alpha=0.5)
                ax.legend()
                ax.grid()
