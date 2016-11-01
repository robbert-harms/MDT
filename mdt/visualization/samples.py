import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from scipy.stats import norm
import matplotlib.mlab as mlab

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class SampleVisualizer(object):

    def __init__(self, voxels):
        self._voxels = voxels
        self.voxel_ind = 0
        self.maps_to_show = sorted(self._voxels.keys())
        self.names = {}
        self._figure = None
        self.show_sliders = True
        self._max_voxel_ind = 0
        self._updating_sliders = False
        self._voxel_slider = None
        self._show_trace = True
        self._nmr_bins = 30
        self._show_slider = True
        self._fit_gaussian = True

    def show(self, voxel_ind=0, names=None, maps_to_show=None, to_file=None, block=True, maximize=False,
             show_trace=True, nmr_bins=20, window_title=None, show_sliders=True, fit_gaussian=True,
             figure_options=None):
        """Show the samples per voxel.

        Args:
            voxel_ind (int): the voxel to show the samples from.
            names (dict): A list of names for the different maps. Use as ``{map_name: display_name}`` that is,
                 the key is the name of the map in the volumes dictionary and the display name is the string that will
                 be used as title for that map.
            maps_to_show (:class:`list`): A list of maps to show.
                The items in this list must correspond to the keys in the volumes dictionary.
            to_file (string, optional, default None): If to_file is not None it is supposed
                to be a filename where the image will be saved.
                If not set to None, nothing will be displayed, the results will directly be saved.
                Already existing items will be overwritten.
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            show_trace (boolean): if we show the trace of each map or not
            nmr_bins (dict or int): either a single value or one per map name
            show_sliders (boolean): if we show the slider or not
            fit_gaussian (boolean): if we fit and show a normal distribution (Gaussian) to the histogram or not
            window_title (str): the title of the window. If None, the default title is used
            figure_options (dict) options for the figure
        """
        figure_options = figure_options or {'figsize': (18, 16)}
        self._figure = plt.figure(**figure_options)

        if names:
            self.names = names
        if maps_to_show:
            self.maps_to_show = maps_to_show
        self.voxel_ind = voxel_ind
        self._nmr_bins = nmr_bins or self._nmr_bins
        self._show_trace = show_trace
        self.show_sliders = show_sliders
        self._fit_gaussian = fit_gaussian

        self._setup()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        if to_file:
            plt.savefig(to_file)
            plt.close()
        else:
            plt.draw()
            if block:
                plt.show(True)

    def set_voxel(self, voxel_ind):
        voxel_ind = round(voxel_ind)
        if not self._updating_sliders:
            self._updating_sliders = True
            self.voxel_ind = int(round(voxel_ind))

            self._voxel_slider.set_val(voxel_ind)
            self._rerender()
            self._voxel_slider.set_val(voxel_ind)
            self._updating_sliders = False

    def _setup(self):
        self._rerender()

        self._max_voxel_ind = max([self._voxels[map_name].shape[0] for map_name in self.maps_to_show])

        y_positions = [0.008]

        if self.show_sliders:
            ax = self._figure.add_axes([0.25, y_positions[0], 0.5, 0.01], axisbg='Wheat')
            self._voxel_slider = _DiscreteSlider(ax, 'Voxel', 0, self._max_voxel_ind - 1,
                                                     valinit=self.voxel_ind, valfmt='%i',
                                                     color='DarkSeaGreen', closedmin=True, closedmax=True)
            self._voxel_slider.on_changed(self.set_voxel)

    def _rerender(self):
        nmr_maps = len(self.maps_to_show)
        if self._show_trace:
            nmr_maps *= 2

        grid = GridSpec(nmr_maps, 1, left=0.04, right=0.96, top=0.94, bottom=0.06, hspace=0.2)

        i = 0
        for map_name in self.maps_to_show:
            samples = self._voxels[map_name]

            title = map_name
            if map_name in self.names:
                title = self.names[map_name]

            if isinstance(self._nmr_bins, dict) and map_name in self._nmr_bins:
                nmr_bins = self._nmr_bins[map_name]
            else:
                nmr_bins = self._nmr_bins

            hist_plot = plt.subplot(grid[i])
            n, bins, patches = hist_plot.hist(samples[self.voxel_ind, :], nmr_bins, normed=True)
            plt.title(title)
            i += 1

            if self._fit_gaussian:
                mu, sigma = norm.fit(samples[self.voxel_ind, :])
                bincenters = 0.5*(bins[1:] + bins[:-1])
                y = mlab.normpdf(bincenters, mu, sigma)
                hist_plot.plot(bincenters, y, 'r', linewidth=1)

            if self._show_trace:
                trace_plot = plt.subplot(grid[i])
                trace_plot.plot(samples[self.voxel_ind, :])
                i += 1


class _DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" and kwarg.

        Args:
            increment (float): specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.25)
        Slider.__init__(self, *args, **kwargs)

    def set_max(self, new_max):
        orig_val = self.val
        self.set_val(self.valmin)

        self.valmax = new_max
        self.ax.set_xlim((self.valmin, self.valmax))

        if orig_val >= new_max:
            self.set_val((new_max + self.valmin) / 2.0)
        else:
            self.set_val(orig_val)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(discrete_val)
