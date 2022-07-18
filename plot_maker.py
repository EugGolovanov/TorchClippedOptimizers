import matplotlib.pyplot as plt


class PlotMaker:
    """
    Класс для отрисовки графиков функций ошибки и метриков моделей
    >>> plot_maker = PlotMaker("Test title", ["first optimizer", "second_optimizer", "third optimizer"], metric_name="IoU", loss_name="focal_loss")
    >>> plot_maker.draw_plot(history, history, history)
    """

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

        fontdict = {'fontsize': 14, 'fontweight': 'medium'}
        ax1.set_title(f"Train loss {self.loss_name}", fontdict=fontdict)
        ax2.set_title(f"Train {self.metric_name}", fontdict=fontdict)
        ax3.set_title(f"Valid {self.metric_name}", fontdict=fontdict)

        for i in range(len(train_losses)):
            ax1.plot(train_losses[i], label=self.names_optimizers[i], alpha=0.5)
        ax1.legend()
        ax1.grid()

        for i in range(len(train_acc)):
            ax2.plot(train_acc[i], label=self.names_optimizers[i], alpha=0.5)
        ax2.legend()
        ax2.grid()

        for i in range(len(val_acc)):
            ax3.plot(val_acc[i], label=self.names_optimizers[i], alpha=0.5)
        ax3.legend()
        ax3.grid()
