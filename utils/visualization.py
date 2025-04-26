import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class DateColorMap(object):

    def __init__(self, ts_length: int, color_map: str = 'Reds'):
        self.ts_length = ts_length
        default_cmap = cm.get_cmap(color_map, ts_length - 1)
        dates_colors = default_cmap(np.linspace(0, 1, ts_length - 1))
        no_change_color = np.array([0, 0, 0, 1])
        cmap_colors = np.zeros((ts_length, 4))
        cmap_colors[0, :] = no_change_color
        cmap_colors[1:, :] = dates_colors
        self.cmap = mpl.colors.ListedColormap(cmap_colors)

    def get_cmap(self):
        return self.cmap

    def get_vmin(self):
        return 0

    def get_vmax(self):
        return self.ts_length


def seg2ch(seg: np.ndarray) -> np.ndarray:
    T, H, W = seg.shape
    ch = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, T):
        prev_seg, current_seg = seg[i - 1], seg[i]
        ch[np.array(prev_seg != current_seg)] = i
    return ch


def plot_continuous_change(ax, ch: np.ndarray, n: int = 5):
    cmap = DateColorMap(n)
    ax.imshow(ch, cmap=cmap.get_cmap(), vmin=cmap.get_vmin(), vmax=cmap.get_vmax(), interpolation=None)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mrf_analysis(ax, y_hat_seg: np.ndarray, y_hat_ch: np.ndarray, map_gt: np.ndarray, t_ch: int, n: int = 5):
    # Example dataset 1 (replace this with your own data)
    criteria = map_gt == t_ch

    y_hat_seg_subset = y_hat_seg[:, criteria]
    y_hat_seg_mean = np.mean(y_hat_seg_subset, axis=-1)
    y_hat_seg_std = np.std(y_hat_seg_subset, axis=-1)

    # Example dataset 2 (replace this with your own data)
    y_hat_ch_subset = y_hat_ch[:-1, criteria]
    y_hat_ch_mean = np.mean(y_hat_ch_subset, axis=-1)
    y_hat_ch_std = np.std(y_hat_ch_subset, axis=-1)

    # Plotting the mean of dataset 1 as a solid line
    x_values_seg = np.arange(0, n)
    ax.plot(x_values_seg, y_hat_seg_mean, color='blue', label='Mean seg')

    # Filling the area for ±1 standard deviation of dataset 1
    ax.fill_between(x_values_seg, y_hat_seg_mean - y_hat_seg_std, y_hat_seg_mean + y_hat_seg_std, color='lightblue',
                     alpha=0.5, label='±1 Std Dev seg')

    # Plotting the vertical lines for dataset 1
    for x_val in x_values_seg:
        plt.vlines(x=x_val, ymin=0, ymax=1, linewidth=1, color='gray')

    # Plotting the bar plots for dataset 2
    x_values_ch = np.arange(0.5, n - 1)
    ax.bar(x_values_ch, y_hat_ch_mean, width=0.4, color='red', alpha=0.6, label='Mean ch ±1 Std Dev')

    # Plotting the vertical lines denoting standard deviation for dataset 2
    for x_val, mean_val, std_val in zip(x_values_ch, y_hat_ch_mean, y_hat_ch_std):
        ax.vlines(x=x_val, ymin=mean_val - std_val, ymax=mean_val + std_val, linewidth=2, color='black')

    # Setting x and y-axis limits
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1)

    # Labeling the axes
    ax.set_xlabel('t')
    ax.set_ylabel('Network output')

    # Title for the plot
    if t_ch == 0:
        ax.set_title(f'Network outputs (seg and ch) for no change')
    else:
        ax.set_title(f'Network outputs (seg and ch) for change between t{t_ch - 1} and t{t_ch}')

    # Display the legend
    ax.legend()


def plot_confusion_matrix(ax, confusion_matrix: np.ndarray, n: int = 5):

    # Define the class labels
    class_labels = ['NC'] + [f't{t-1}-t{t}' for t in range(1, n)]

    # Plot the confusion matrix with colors
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    # Add color bar legend
    # ax.colorbar(cax)
    ax.figure.colorbar(cax, ax=ax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    # Loop over data dimensions and create text annotations
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, f'{confusion_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    # Set title and axis labels
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix (Relative Frequencies)')

