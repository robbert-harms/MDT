import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mdt import get_slice_in_dimension
from mdt.visualization.maps.base import Clipping, Scale
from mdt.visualization.utils import MyColourBarTickLocator

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapsVisualizer(object):

    def __init__(self, data_info, figure):
        self._data_info = data_info
        self._figure = figure

    def render(self, plot_config):
        """Render all the maps to the figure. This is for use in GUI embedded situations."""
        Renderer(self._data_info, self._figure, plot_config).render()

    def to_file(self, file_name, plot_config, **kwargs):
        """Renders the figures to the given filename."""
        Renderer(self._data_info, self._figure, plot_config).render()

        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        kwargs['dpi'] = kwargs.get('dpi') or 300
        self._figure.savefig(file_name, **kwargs)

    def show(self, plot_config, block=True, maximize=False, window_title=None):
        """Show the data contained in this visualizer using the specifics in this function call.

        Args:
            plot_config (mdt.visualization.maps.base.MapPlotConfig): the plot configuration
            block (boolean): If we want to block after calling the plots or not. Set this to False if you
                do not want the routine to block after drawing. In doing so you manually need to block.
            maximize (boolean): if we want to display the window maximized or not
            window_title (str): the title of the window. If None, the default title is used
        """
        Renderer(self._data_info, self._figure, plot_config).render()

        if maximize:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()

        if window_title:
            mng = plt.get_current_fig_manager()
            mng.canvas.set_window_title(window_title)

        if block:
            plt.show(True)


class Renderer(object):

    def __init__(self, data_info, figure, plot_config):
        """Create a new renderer for the given volumes on the given figure using the given configuration.

        Args:
            data_info (DataInfo): the information about the maps to show
            figure (Figure): the matplotlib figure to draw on
            plot_config (mdt.visualization.maps.base.MapPlotConfig): the plot configuration
        """
        self._data_info = data_info
        self._figure = figure
        self._plot_config = plot_config

    def render(self):
        """Render the maps"""
        grid_layout_specifier = self._plot_config.grid_layout.get_gridspec(
            self._figure, len(self._plot_config.maps_to_show))

        for ind, map_name in enumerate(self._plot_config.maps_to_show):
            axis = grid_layout_specifier.get_axis(ind)
            self._render_map(map_name, axis)

    def _render_map(self, map_name, axis):
        axis.set_title(self._get_title(map_name))
        axis.axis('on' if self._plot_config.show_axis else 'off')

        data = self._get_image(map_name)
        if self._plot_config.rotate:
            data = np.rot90(data, self._plot_config.rotate // 90)
        data = self._get_map_specific_option(map_name, 'clipping', Clipping()).apply(data)
        data = self._plot_config.zoom.apply(data)

        plot_options = self._get_map_plot_options(map_name)
        vf = axis.imshow(data, **plot_options)

        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)

        self._add_colorbar(map_name, colorbar_axis, vf)
        self._apply_font(axis, colorbar_axis)

    def _apply_font(self, image_axis, colorbar_axis):
        items = [image_axis.xaxis.label, image_axis.yaxis.label]
        items.extend(image_axis.get_xticklabels())
        items.extend(image_axis.get_yticklabels())
        items.extend(colorbar_axis.yaxis.get_ticklabels())

        for item in items:
            item.set_fontsize(self._plot_config.font.size - 2)
            item.set_family(self._plot_config.font.name)

        image_axis.title.set_fontsize(self._plot_config.font.size)
        image_axis.title.set_family(self._plot_config.font.name)

        colorbar_axis.yaxis.offsetText.set_fontsize(self._plot_config.font.size - 3)
        colorbar_axis.yaxis.offsetText.set_family(self._plot_config.font.name)

    def _add_colorbar(self, map_name, axis, image_figure):
        cbar = plt.colorbar(image_figure, cax=axis, ticks=self._get_tick_locator(map_name))
        cbar.formatter.set_powerlimits((-3, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        if cbar.ax.get_yticklabels():
            cbar.ax.get_yticklabels()[-1].set_verticalalignment('top')
        return cbar

    def _get_map_specific_option(self, map_name, option, default):
        if map_name in self._plot_config.map_plot_options:
            value = getattr(self._plot_config.map_plot_options[map_name], option)
            if value:
                return value
        return default

    def _get_title(self, map_name):
        return self._get_map_specific_option(map_name, 'title', map_name)

    def _get_map_plot_options(self, map_name):
        output_dict = {'vmin': self._data_info.maps[map_name].min(),
                       'vmax': self._data_info.maps[map_name].max(),
                       'cmap': self._get_map_specific_option(map_name, 'colormap', self._plot_config.colormap)}

        scale = self._get_map_specific_option(map_name, 'scale', Scale())
        if scale.vmax is not None:
            output_dict['vmax'] = scale.vmax
        if scale.vmin is not None:
            output_dict['vmin'] = scale.vmin

        return output_dict

    def _get_image(self, map_name):
        """Get the 2d image to display for the given data."""
        data = self._data_info.maps[map_name]
        dimension = self._plot_config.dimension
        slice_index = self._plot_config.slice_index
        volume_index = self._plot_config.volume_index

        data = get_slice_in_dimension(data, dimension, slice_index)
        if len(data.shape) > 2:
            data = np.squeeze(data[:, :, volume_index])
        return data

    def _get_tick_locator(self, map_name):
        min_val, max_val = self._data_info.maps[map_name].min(), self._data_info.maps[map_name].max()
        return MyColourBarTickLocator(min_val, max_val, numticks=self._plot_config.colorbar_nmr_ticks)
