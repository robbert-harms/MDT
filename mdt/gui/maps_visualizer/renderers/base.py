__author__ = 'Robbert Harms'
__date__ = '2017-06-19'
__maintainer__ = 'Robbert Harms'
__email__ = 'robbert.harms@maastrichtuniversity.nl'
__licence__ = 'LGPL v3'


class PlottingFrame(object):

    def __init__(self, controller, plotting_info_viewer=None):
        super(PlottingFrame, self).__init__()
        self._controller = controller
        self._plotting_info_viewer = plotting_info_viewer or NoOptPlottingFrameInfoViewer()

    def set_auto_rendering(self, auto_render):
        """Set if this plotting frame should auto render itself on every configuration update, or not.

        Args:
            auto_render (boolean): if True the plotting frame should auto render, if False it should only
                render on manual updates.
        """

    def redraw(self):
        """Tell the plotting frame to do a redraw."""

    def export_image(self, filename, width, height, dpi=100):
        """Export the current view as an image.

        Args:
            filename (str): where to write the file
            width (int): the width in pixels
            height (int): the height in pixels
            dpi (int): the dpi of the result
        """


class PlottingFrameInfoViewer(object):

    def __init__(self):
        """Implementations of this class can be given to a PlottingFrame to update viewing information.

        As an interface is bridges the gap between the rest of the GUI and the PlottingFrame and
        can encapsulate highlighting interesting aspects of one of the plots.
        """

    def set_voxel_info(self, onscreen_coords, data_index, value):
        """Highlight a single voxel.

        Args:
            onscreen_coords (tuple of x,y): the coordinates of the voxel onscreen
            data_index (tuple of x,y,z,v): the 4d coordinates of the corresponding voxel in the data
            value (float): the value of the object in the 4d coordinates.
        """

    def clear_voxel_info(self):
        """Tell the info viewer that we are no longer looking at a specific voxel."""


class NoOptPlottingFrameInfoViewer(PlottingFrameInfoViewer):

    def set_voxel_info(self, onscreen_coords, data_index, value):
        super(NoOptPlottingFrameInfoViewer, self).set_voxel_info(onscreen_coords, data_index, value)

    def clear_voxel_info(self):
        super(NoOptPlottingFrameInfoViewer, self).clear_voxel_info()
