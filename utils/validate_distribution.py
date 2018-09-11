from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import os.path
import torch


class ValObj:
    def __init__(self, base_path, min=-1.11, max=100):
        self.coarse_pr = torch.load(base_path + '_coarse_pr.pt')
        self.fine_pr = torch.load(base_path + '_fine_pr.pt')
        self.recon_pr = torch.load(base_path + '_recon_pr.pt')
        self.min = min
        self.max = max
        self.bins = np.arange(self.min, self.max, 0.005)
        self.density = True
        self.cumulative = True

    def set_bins(self):
        self.bins = np.arange(self.min, self.max, 0.005)

    def plot_hist(self,ax, i, j):
        ax.hist(self.recon_pr[:, 0, i, j].detach().numpy(), bins=self.bins, histtype='step',
                cumulative=self.cumulative,
                density=self.density,
                label='recon_pr')
        ax.hist(self.fine_pr[:, 0, i, j].detach().numpy(), bins=self.bins, histtype='step',
                cumulative=self.cumulative,
                density=self.density,
                label='fine_pr')
        ax.hist(self.coarse_pr[:, 0, 0, 0].detach().numpy(), bins=self.bins, histtype='step',
                cumulative=self.cumulative,
                label='coarse_pr', density=self.density)
        ax.legend()

    def evaluate_distribution(self):
        fig2, ax = plt.subplots()
        n_sliders = 4
        # Create offset for sliders
        offset = 5*n_sliders/100
        plt.subplots_adjust(bottom=0.15 + offset)
        # Plot with initial slider postition
        self.plot_hist(ax, 3, 3)

        # Create Sliders
        ij_axis = [plt.axes([0.25, 0.05 + i_offset, 0.65, 0.03]) for i_offset in np.arange(offset, 0.0, -0.05)]
        ij_sliders = [Slider(ij_axis[i], ['i', 'j'][i], 0, 7, valinit=3, valstep=1) for i in range(2)]
        bin_slider_min = Slider(ij_axis[-2], ['Bins Start'], -1.11, 150, valinit=self.min)
        bin_slider_max = Slider(ij_axis[-1], ['Bins Stop'], -1.11, 150, valinit=self.max)

        def update(val):
            ax.cla()
            i = int(ij_sliders[0].val)
            j = int(ij_sliders[1].val)
            self.plot_hist(ax, i, j)
            fig2.canvas.draw_idle()
            plt.draw()

        def update_min(val):
            self.min = bin_slider_min.val
            self.set_bins()
            update(val)

        def update_max(val):
            self.max = bin_slider_max.val
            self.set_bins()
            update(val)

        for slider in ij_sliders:
            slider.on_changed(update)

        bin_slider_min.on_changed(update_min)
        bin_slider_max.on_changed(update_max)

        plt.show()


def save_val_data(all_fine_pr, all_coarse, all_recon_pr, base_path):
    torch.save(all_coarse, base_path + '_coarse_pr.pt')
    torch.save(all_fine_pr, base_path + '_fine_pr.pt')
    torch.save(all_recon_pr, base_path + '_recon_pr.pt')