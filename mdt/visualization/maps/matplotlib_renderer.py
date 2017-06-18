import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mdt import get_slice_in_dimension
from mdt.visualization.maps.base import Clipping, Scale, Point
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
        """Render all the maps to the figure. This is for use in GUI embedded situations.

        Returns:
            list of AxisData: the list with the drawn axes and the accompanying data
        """
        renderer = Renderer(self._data_info, self._figure, plot_config)
        renderer.render()
        return renderer.image_axes

    def to_file(self, file_name, plot_config, **kwargs):
        """Renders the figures to the given filename."""
        Renderer(self._data_info, self._figure, plot_config).render()

        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        kwargs['dpi'] = kwargs.get('dpi') or 100
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


class AxisData(object):

    def __init__(self, axis, map_name, map_info, plot_config):
        """Contains a reference to a drawn matpotlib axis and to the accompanying data.

        Args:
            axis (Axis): the matpotlib axis
            map_name (str): the name/key of this map
            map_info (SingleMapInfo): the map information
            plot_config (MapPlotConfig): the map plot configuration
        """
        self.axis = axis
        self.map_name = map_name
        self._map_info = map_info
        self._plot_config = plot_config

    def coordinates_to_index(self, x, y):
        """Converts data coordinates to index coordinates of the array.

        Args:
            x (int): The x-coordinate in data coordinates.
            y (int): The y-coordinate in data coordinates.

        Returns
            tuple: Index coordinates of the map associated with the image (x, y, z, d).
        """
        return _coordinates_to_index(self._map_info, self._plot_config, x, y)

    def get_value(self, index):
        """Get the value of this axis data at the given index.

        Args:
            index (tuple)): the 3d or 4d index to the map corresponding to this axis data (x, y, z, [v])

        Returns:
            float: the value at the given index.
        """
        return self._map_info.data[tuple(index)]


class Renderer(object):

    def __init__(self, data_info, figure, plot_config):
        """Create a new renderer for the given volumes on the given figure using the given configuration.

        This renders the images with flipped upside down with the origin at the bottom left. The upside down flip
        is necessary to allow counter-clockwise rotation.

        Args:
            data_info (DataInfo): the information about the maps to show
            figure (Figure): the matplotlib figure to draw on
            plot_config (mdt.visualization.maps.base.MapPlotConfig): the plot configuration
        """
        self._data_info = data_info
        self._figure = figure
        self._plot_config = plot_config
        self.image_axes = []

    def render(self):
        """Render the maps"""
        grid_layout_specifier = self._plot_config.grid_layout.get_gridspec(
            self._figure, len(self._plot_config.maps_to_show))

        if self._plot_config.title:
            self._figure.suptitle(self._plot_config.title, fontsize=self._plot_config.font.size,
                                  family=self._plot_config.font.name)

        for ind, map_name in enumerate(self._plot_config.maps_to_show):
            axis = grid_layout_specifier.get_axis(ind)
            axis_data = self._render_map(map_name, axis)
            self.image_axes.append(axis_data)

    def _render_map(self, map_name, axis):
        """Render a single map to the given axis"""
        axis.set_title(self._get_title(map_name), y=self._get_title_spacing(map_name))
        axis.axis('on' if self._plot_config.show_axis else 'off')

        data = self._get_image(map_name)
        if self._plot_config.rotate:
            data = np.rot90(data, self._plot_config.rotate // 90)

        if not self._plot_config.flipud:
            # by default we flipud to correct for matplotlib lower origin. If the user
            # sets flipud, we do not need to to it
            data = np.flipud(data)

        data = self._plot_config.zoom.apply(data)

        plot_options = self._get_map_plot_options(map_name)
        plot_options['origin'] = 'lower'
        plot_options['interpolation'] = self._plot_config.interpolation
        vf = axis.imshow(data, **plot_options)

        self._add_patches(map_name, axis)

        if self._get_map_attr(map_name, 'show_colorbar', self._plot_config.show_colorbar):
            divider = make_axes_locatable(axis)
            colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)

            self._add_colorbar(map_name, colorbar_axis, vf, self._get_map_attr(map_name, 'colorbar_label'))
            self._apply_font(axis, colorbar_axis)

        return AxisData(axis, map_name, self._data_info.get_single_map_info(map_name), self._plot_config)

    def _apply_font(self, image_axis, colorbar_axis):
        """Apply the font from the plot configuration to the image and colorbar axis"""
        items = [image_axis.xaxis.label, image_axis.yaxis.label]
        items.extend(image_axis.get_xticklabels())
        items.extend(image_axis.get_yticklabels())
        items.extend(colorbar_axis.yaxis.get_ticklabels())

        for item in items:
            item.set_fontsize(self._plot_config.font.size - 2)
            item.set_family(self._plot_config.font.name)

        image_axis.title.set_fontsize(self._plot_config.font.size)
        image_axis.title.set_family(self._plot_config.font.name)

        colorbar_axis.yaxis.label.set_fontsize(self._plot_config.font.size)
        colorbar_axis.yaxis.label.set_family(self._plot_config.font.name)

        colorbar_axis.yaxis.offsetText.set_fontsize(self._plot_config.font.size - 3)
        colorbar_axis.yaxis.offsetText.set_family(self._plot_config.font.name)

    def _add_patches(self, map_name, axis):
        """Add the patches defined in the global config and in the map specific config to this image plot."""
        for patch_info in self._plot_config.drawable_patches:
            axis.add_patch(patch_info.get_patch())

        for patch_info in self._get_map_attr(map_name, 'drawable_patches', []):
            axis.add_patch(patch_info.get_patch())

    def _add_colorbar(self, map_name, axis, image_figure, colorbar_label):
        """Add a colorbar to the axis

        Returns:
            axis: the colorbar axis
        """
        kwargs = dict(cax=axis, ticks=self._get_tick_locator(map_name))
        if colorbar_label:
            kwargs.update(dict(label=colorbar_label))

        cbar = plt.colorbar(image_figure, **kwargs)
        cbar.formatter.set_powerlimits((-3, 4))
        cbar.ax.yaxis.set_offset_position('left')
        cbar.update_ticks()

        if cbar.ax.get_yticklabels():
            cbar.ax.get_yticklabels()[-1].set_verticalalignment('top')
        return cbar

    def _get_map_attr(self, map_name, option, default=None):
        if map_name in self._plot_config.map_plot_options:
            value = getattr(self._plot_config.map_plot_options[map_name], option)
            if value is not None:
                return value
        return default

    def _get_title(self, map_name):
        return self._get_map_attr(map_name, 'title', map_name)

    def _get_title_spacing(self, map_name):
        return 1 + self._get_map_attr(map_name, 'title_spacing', 0)

    def _get_map_plot_options(self, map_name):
        output_dict = {'vmin': self._data_info.get_single_map_info(map_name).min(),
                       'vmax': self._data_info.get_single_map_info(map_name).max(),
                       'cmap': self._get_map_attr(map_name, 'colormap', self._plot_config.colormap)}

        scale = self._get_map_attr(map_name, 'scale', Scale())
        if scale.use_max:
            output_dict['vmax'] = scale.vmax
        if scale.use_min:
            output_dict['vmin'] = scale.vmin

        return output_dict

    def _get_image(self, map_name):
        """Get the 2d image to display for the given data."""
        data = self._data_info.get_map_data(map_name)

        dimension = self._plot_config.dimension
        slice_index = self._plot_config.slice_index
        volume_index = self._plot_config.volume_index

        def get_slice(data):
            data_slice = get_slice_in_dimension(data, dimension, slice_index)
            if len(data_slice.shape) > 2:
                if volume_index < data_slice.shape[2]:
                    data_slice = np.squeeze(data_slice[:, :, volume_index])
                else:
                    data_slice = np.squeeze(data_slice[:, :, data_slice.shape[2] - 1])

            if len(data_slice.shape) == 1:
                data_slice = data_slice[:, None]

            return data_slice

        if len(data.shape) == 2:
            data_slice = data
        else:
            data_slice = get_slice(data)

        data_slice = self._get_map_attr(map_name, 'clipping', Clipping()).apply(data_slice)

        mask_name = self._get_map_attr(map_name, 'mask_name', self._plot_config.mask_name)
        if mask_name:
            data_slice = data_slice * (get_slice(self._data_info.get_map_data(mask_name)) > 0)

        return data_slice

    def _get_tick_locator(self, map_name):
        min_val, max_val = self._data_info.get_single_map_info(map_name).min_max()

        scale = self._get_map_attr(map_name, 'scale', Scale())
        if scale.use_max:
            max_val = scale.vmax
        if scale.use_min:
            min_val = scale.vmin

        return MyColourBarTickLocator(min_val, max_val, numticks=self._plot_config.colorbar_nmr_ticks)


def _coordinates_to_index(map_info, plot_config, x, y):
    """Converts data coordinates to index coordinates of the array.

    Args:
        map_info (SingleMapInfo): the map information
        plot_config (MapPlotConfig): the map plot configuration
        x (int): The x-coordinate in data coordinates.
        y (int): The y-coordinate in data coordinates.

    Returns
        tuple: Index coordinates of the map associated with the image (x, y, z, d).
    """
    shape = map_info.get_size_in_dimension(plot_config.dimension, plot_config.rotate)

    # correct for zoom
    x += plot_config.zoom.p0.x
    y += plot_config.zoom.p0.y

    # correct for flip upside down
    if not plot_config.flipud:
        y = map_info.get_max_y_index(plot_config.dimension, plot_config.rotate) - y

    # correct for displayed axis, the view is x-data on y-image and y-data on x-image
    x, y = y, x

    # rotate the point
    rotated = Point(x, y).rotate90((-1 * plot_config.rotate % 360) // 90)

    # translate the point back to a new origin
    if plot_config.rotate == 90:
        rotated.y = shape[1] + rotated.y
    elif plot_config.rotate == 180:
        rotated.x = shape[1] + rotated.x
        rotated.y = shape[0] + rotated.y
    elif plot_config.rotate == 270:
        rotated.x = shape[0] + rotated.x

    if len(map_info.shape) == 2:
        return [rotated.x, rotated.y]

    # create the index
    index = [rotated.x, rotated.y]
    index.insert(plot_config.dimension, plot_config.slice_index)

    if len(map_info.data.shape) > 3:
        if plot_config.volume_index < map_info.data.shape[3]:
            index.append(plot_config.volume_index)
        else:
            index.append(map_info.data.shape[3] - 1)

    return index

