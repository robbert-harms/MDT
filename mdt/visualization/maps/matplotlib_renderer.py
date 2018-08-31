import os
import numpy as np
from matplotlib import pyplot as plt, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mdt import get_slice_in_dimension
from mdt.visualization.maps.base import Clipping, Scale, Point2d
from mdt.visualization.utils import MyColourBarTickLocator

__author__ = 'Robbert Harms'
__date__ = "2016-09-02"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class MapsVisualizer:

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

        if not os.path.dirname(file_name).strip():
            return

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


class AxisData:

    def __init__(self, axis, map_name, map_info, plot_config):
        """Contains a reference to a drawn matpotlib axis and to the accompanying data.

        Args:
            axis (Axis): the matpotlib axis
            map_name (str): the name/key of this map
            map_info (mdt.visualization.maps.base.SingleMapInfo): the map information
            plot_config (mdt.visualization.maps.base.MapPlotConfig): the map plot configuration
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


class Renderer:

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
        if self._get_map_attr(map_name, 'show_title', self._plot_config.show_titles):
            axis.set_title(self._get_map_attr(map_name, 'title', map_name), y=self._get_title_spacing(map_name))

        axis.axis('on' if self._plot_config.show_axis else 'off')
        data = self._get_image(map_name)

        plot_options = self._get_map_plot_options(map_name)
        plot_options['origin'] = 'lower'
        plot_options['interpolation'] = self._plot_config.interpolation
        vf = axis.imshow(data, **plot_options)

        self._add_annotations(map_name, axis)

        self._add_colorbar(axis, map_name, vf, self._get_map_attr(map_name, 'colorbar_label'))

        self._apply_font_general_axis(axis)
        return AxisData(axis, map_name, self._data_info.get_single_map_info(map_name), self._plot_config)

    def _apply_font_general_axis(self, image_axis):
        """Apply the font from the plot configuration to the image axis"""
        items = [image_axis.xaxis.label, image_axis.yaxis.label]
        items.extend(image_axis.get_xticklabels())
        items.extend(image_axis.get_yticklabels())

        for item in items:
            item.set_fontsize(self._plot_config.font.size - 2)
            item.set_family(self._plot_config.font.name)

        image_axis.title.set_fontsize(self._plot_config.font.size)
        image_axis.title.set_family(self._plot_config.font.name)

    def _apply_font_colorbar_axis(self, colorbar_axis):
        """Apply the font from the plot configuration to the colorbar axis"""
        for axes in [colorbar_axis.xaxis, colorbar_axis.yaxis]:
            items = axes.get_ticklabels()

            for item in items:
                item.set_fontsize(self._plot_config.font.size - 2)
                item.set_family(self._plot_config.font.name)

            axes.label.set_fontsize(self._plot_config.font.size)
            axes.label.set_family(self._plot_config.font.name)

            axes.offsetText.set_fontsize(self._plot_config.font.size - 3)
            axes.offsetText.set_family(self._plot_config.font.name)

    def _add_annotations(self, map_name, axis):
        def get_value(annotation):
            data = self._data_info.get_map_data(map_name)
            index = tuple(annotation.voxel_index)

            if len(data.shape) > 3 and data.shape[3] > 1:
                if len(index) < 4:
                    index = index + (self._plot_config.volume_index,)
                return float(data[index])
            return float(data[index[:3]])

        def get_font_size(annotation):
            font_size = self._plot_config.font.size - 3
            if annotation.font_size is not None:
                font_size = annotation.font_size
            return font_size

        def get_coordinate(annotation):
            coordinate = _index_to_coordinates(self._data_info.get_single_map_info(map_name),
                                               self._plot_config, annotation.voxel_index)
            if coordinate is None:
                return None
            return np.array(coordinate)

        def get_text_box_location(annotation):
            axis_to_data = axis.transData + axis.transAxes.inverted()

            coordinate_normalized = axis_to_data.transform(get_coordinate(annotation))

            delta = annotation.text_distance + axis_to_data.transform([annotation.marker_size] * 2)[0]

            definitions = {
                'upper left': [(-delta, +delta), 'right', 'bottom'],
                'top left': [(-delta, +delta), 'right', 'bottom'],
                'upper right': [(+delta, +delta), 'left', 'bottom'],
                'top right': [(+delta, +delta), 'left', 'bottom'],

                'lower left': [(-delta, -delta), 'right', 'top'],
                'bottom left': [(-delta, -delta), 'right', 'top'],
                'lower right': [(+delta, -delta), 'left', 'top'],
                'bottom right': [(+delta, -delta), 'left', 'top'],

                'top': [(0, +np.sqrt(2) * delta), 'center', 'bottom'],
                'north': [(0, +np.sqrt(2) * delta), 'center', 'bottom'],

                'bottom': [(0, -np.sqrt(2) * delta), 'center', 'top'],
                'south': [(0, -np.sqrt(2) * delta), 'center', 'top'],

                'left': [(-np.sqrt(2) * delta, 0), 'right', 'center'],
                'west': [(-np.sqrt(2) * delta, 0), 'right', 'center'],

                'right': [(+np.sqrt(2) * delta, 0), 'left', 'center'],
                'east': [(+np.sqrt(2) * delta, 0), 'left', 'center'],
            }

            transform, halign, valign = definitions[annotation.text_location]
            return axis_to_data.inverted().transform(coordinate_normalized + transform), \
                   halign, valign

        def get_arrow_head_location(annotation):
            coordinate = get_coordinate(annotation)
            delta = annotation.marker_size / 2.
            definitions = {
                'upper left': (-delta, +delta),
                'top left': (-delta, +delta),
                'upper right': (+delta, +delta),
                'top right': (+delta, +delta),

                'lower left': (-delta, -delta),
                'bottom left': (-delta, -delta),
                'lower right': (+delta, -delta),
                'bottom right': (+delta, -delta),

                'top': (0, delta),
                'north': (0, delta),

                'bottom': (0, -delta),
                'south': (0, -delta),

                'left': (-delta, 0),
                'west': (-delta, 0),

                'right': (delta, 0),
                'east': (delta, 0),
            }

            return coordinate + definitions[annotation.text_location]

        def draw_annotation(annotation):
            coordinate = get_coordinate(annotation)
            if coordinate is None:
                return

            axis.add_patch(patches.Rectangle(coordinate - (annotation.marker_size / 2.),
                                             annotation.marker_size, annotation.marker_size,
                                             linewidth=1, edgecolor='white', facecolor='#0066ff'))

            xy_text, horizontal_alignment, vertical_alignment = get_text_box_location(annotation)

            axis.annotate(
                annotation.text_template.format(voxel_index=tuple(annotation.voxel_index), value=get_value(annotation)),
                xy=get_arrow_head_location(annotation),
                xytext=xy_text,
                horizontalalignment=horizontal_alignment, verticalalignment=vertical_alignment,
                multialignment='left',
                arrowprops=dict(color='white', connectionstyle="arc3",
                                width=annotation.arrow_width, headwidth=8 + annotation.arrow_width, headlength=10),
                color='black',
                size=get_font_size(annotation),
                family=self._plot_config.font.name,
                bbox=dict(facecolor='white'))

        for annotation in self._plot_config.annotations:
            draw_annotation(annotation)

    def _get_colorbar_setting(self, map_name, attr_name):
        map_specific_colorbar_settings = self._get_map_attr(map_name, 'colorbar_settings')
        global_colorbar_settings = self._plot_config.colorbar_settings
        return map_specific_colorbar_settings.get_preferred(attr_name, [global_colorbar_settings])

    def _add_colorbar(self, axis, map_name, image_figure, colorbar_label):
        """Add a colorbar to the axis

        Returns:
            axis: the image axis
        """
        divider = make_axes_locatable(axis)

        colorbar_location = self._get_colorbar_setting(map_name, 'location')

        show_colorbar = self._get_colorbar_setting(map_name, 'visible')
        at_least_one_colorbar = any(self._get_colorbar_setting(name, 'visible')
                                    for name in self._plot_config.map_plot_options)

        if at_least_one_colorbar:
            axis_kwargs = dict(size="5%", pad=0.1)

            if self._plot_config.show_axis and colorbar_location in ['bottom', 'left']:
                axis_kwargs['pad'] = 0.3

            if show_colorbar and colorbar_location in ('left', 'right'):
                colorbar_axis = divider.append_axes(colorbar_location, **axis_kwargs)
            else:
                fake_axis = divider.append_axes('right', **axis_kwargs)
                fake_axis.axis('off')

            if show_colorbar and colorbar_location in ('top', 'bottom'):
                colorbar_axis = divider.append_axes(colorbar_location, **axis_kwargs)
            else:
                fake_axis = divider.append_axes('bottom', **axis_kwargs)
                fake_axis.axis('off')

        if show_colorbar:
            kwargs = dict(cax=colorbar_axis, ticks=self._get_tick_locator(map_name))
            if colorbar_label:
                kwargs.update(dict(label=colorbar_label))

            if colorbar_location in ['top', 'bottom']:
                kwargs['orientation'] = 'horizontal'

            cbar = plt.colorbar(image_figure, **kwargs)
            cbar.formatter.set_powerlimits(self._get_colorbar_setting(map_name, 'power_limits'))
            colorbar_axis.yaxis.set_offset_position('left')

            if colorbar_location == 'left':
                colorbar_axis.yaxis.set_ticks_position('left')
                colorbar_axis.yaxis.set_label_position('left')

            if colorbar_location == 'top':
                colorbar_axis.xaxis.set_ticks_position('top')
                colorbar_axis.xaxis.set_label_position('top')

            cbar.update_ticks()

            if cbar.ax.get_yticklabels():
                cbar.ax.get_yticklabels()[-1].set_verticalalignment('top')
            elif cbar.ax.get_xticklabels():
                cbar.ax.get_xticklabels()[0].set_horizontalalignment('left')
                cbar.ax.get_xticklabels()[-1].set_horizontalalignment('right')

            self._apply_font_colorbar_axis(colorbar_axis)

    def _get_map_attr(self, map_name, option, default=None):
        if map_name in self._plot_config.map_plot_options:
            value = getattr(self._plot_config.map_plot_options[map_name], option)
            if value is not None:
                return value
        return default

    def _get_title_spacing(self, map_name):
        spacing = 1 + self._get_map_attr(map_name, 'title_spacing', self._plot_config.title_spacing or 0)
        if self._get_colorbar_setting(map_name, 'location') == 'top':
            spacing += 0.15
        return spacing

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
        def get_slice(data):
            """Get the slice of the data in the desired dimension."""
            dimension = self._plot_config.dimension
            slice_index = self._plot_config.slice_index
            volume_index = self._plot_config.volume_index

            data_slice = get_slice_in_dimension(data, dimension, slice_index)
            if len(data_slice.shape) > 2:
                if data_slice.shape[2] == 3 and self._get_map_attr(map_name, 'interpret_as_colormap'):
                    data_slice = np.abs(data_slice[:, :, :])

                    if self._get_map_attr(map_name, 'colormap_order'):
                        order = self._get_map_attr(map_name, 'colormap_order')
                        data_slice[..., :] = data_slice[..., [order.index(color) for color in 'rbg']]

                elif volume_index < data_slice.shape[2]:
                    data_slice = np.squeeze(data_slice[:, :, volume_index])
                else:
                    data_slice = np.squeeze(data_slice[:, :, data_slice.shape[2] - 1])

            if len(data_slice.shape) == 1:
                data_slice = data_slice[:, None]

            return data_slice

        def apply_modifications(data_slice):
            """Modify the data using (possible) value clipping and masking"""
            data_slice = self._get_map_attr(map_name, 'clipping', Clipping()).apply(data_slice)
            mask_name = self._get_map_attr(map_name, 'mask_name', self._plot_config.mask_name)
            if mask_name:
                mask = (get_slice(self._data_info.get_map_data(mask_name)) > 0)
                if len(data_slice.shape) != len(mask.shape):
                    mask = mask[..., None]
                data_slice = data_slice * mask
            return data_slice

        data = self._data_info.get_map_data(map_name)
        if len(data.shape) == 2:
            data_slice = data
        else:
            data_slice = get_slice(data)

        data_slice = apply_modifications(data_slice)
        data_slice = _apply_transformations(self._plot_config, data_slice)

        if len(data_slice.shape) == 3:
            multiplication_map = self._get_map_attr(map_name, 'colormap_weight_map', None)
            if multiplication_map is not None and map_name != multiplication_map:
                multiplication_map = self._get_image(multiplication_map)
                if len(multiplication_map.shape) != 3:
                    multiplication_map = multiplication_map[..., None]

                scale = self._get_map_attr(map_name, 'scale', Scale())
                new_vmax = np.min(((scale.vmax if scale.use_max else 1), 1))
                new_vmin = np.max(((scale.vmin if scale.use_min else 0), 0))

                multiplication_map = multiplication_map + (1 - (new_vmax - new_vmin))
                data_slice = data_slice * multiplication_map

                if np.max(data_slice) > 1:
                    data_slice /= np.max(data_slice)

        return data_slice

    def _get_tick_locator(self, map_name):
        min_val, max_val = self._data_info.get_single_map_info(map_name).min_max()

        scale = self._get_map_attr(map_name, 'scale', Scale())
        if scale.use_max:
            max_val = scale.vmax
        if scale.use_min:
            min_val = scale.vmin

        return MyColourBarTickLocator(
            min_val, max_val, numticks=self._get_colorbar_setting(map_name, 'nmr_ticks'),
            round_precision=self._get_colorbar_setting(map_name, 'round_precision')
        )


def _apply_transformations(plot_config, data_slice):
    """Rotate, flip and zoom the data slice.

    Depending on the plot configuration, this will apply some transformations to the given data slice.

    Args:
        plot_config (mdt.visualization.maps.base.MapPlotConfig): the plot configuration
        data_slice (ndarray): the 2d slice of data to transform

    Returns:
        ndarray: the transformed 2d slice of data
    """
    if plot_config.rotate:
        data_slice = np.rot90(data_slice, plot_config.rotate // 90)

    if not plot_config.flipud:
        # by default we flipud to correct for matplotlib lower origin. If the user
        # sets flipud, we do not need to to it
        data_slice = np.flipud(data_slice)

    data_slice = plot_config.zoom.apply(data_slice)
    return data_slice


def _coordinates_to_index(map_info, plot_config, x, y):
    """Converts data coordinates to index coordinates of the array.

    This is the inverse of :func:`_index_to_coordinates`.

    Args:
        map_info (mdt.visualization.maps.base.SingleMapInfo): the map information
        plot_config (mdt.visualization.maps.base.MapPlotConfig): the map plot configuration
        x (int): The x-coordinate in data coordinates.
        y (int): The y-coordinate in data coordinates.

    Returns
        tuple: Index coordinates of the map associated with the image (x, y, z, w).
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
    rotated = Point2d(x, y).rotate90((-1 * plot_config.rotate % 360) // 90)

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

    # fixes issue with off-by-one errors
    for ind in range(len(index)):
        if index[ind] >= map_info.shape[ind]:
            index[ind] = map_info.shape[ind] - 1
        if index[ind] < 0:
            index[ind] = 0

    return [int(el) for el in index]


def _index_to_coordinates(map_info, plot_config, index):
    """Converts data index coordinates to an onscreen position.

    This is the inverse of :func:`_coordinates_to_index`.

    Args:
        map_info (mdt.visualization.maps.base.SingleMapInfo): the map information
        plot_config (mdt.visualization.maps.base.MapPlotConfig): the map plot configuration
        index (tuple of int): The data index we wish to translate to an onscreen position.

    Returns
        tuple or None: Coordinates of the given index on the screen (x, y) or None if coordinate not visible.
    """
    dimension = plot_config.dimension

    if index[dimension] != plot_config.slice_index:
        return None

    reduced_index = list(index[:3])
    del reduced_index[dimension]

    img_shape = list(map_info.shape[:3])
    del img_shape[dimension]

    index_slice = np.arange(0, int(np.prod(img_shape)), 1).reshape(img_shape)

    index_nmr = index_slice[tuple(reduced_index)]
    coordinates = np.where(_apply_transformations(plot_config, index_slice) == index_nmr)

    if len(coordinates[0]) and len(coordinates[1]):
        return coordinates[1][0], coordinates[0][0]
    return None
