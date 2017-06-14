import glob
import os

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from mdt import results_preselection_names
from mdt.nifti import load_nifti
from mdt.visualization.maps.base import SimpleDataInfo, MapPlotConfig
from mdt.gui.maps_visualizer.main import start_gui
from mdt.gui.model_fit.design.ui_view_results_tab import Ui_ViewResultsTabContent
from mdt.gui.utils import MainTab
from mdt.utils import split_image_path

__author__ = 'Robbert Harms'
__date__ = "2016-06-27"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class ViewResultsTab(MainTab, Ui_ViewResultsTabContent):

    def __init__(self, shared_state, computations_thread):
        self._shared_state = shared_state
        self._parameter_files = {}
        self._folder = None

    def setupUi(self, ViewResultsTabContent):
        super(ViewResultsTab, self).setupUi(ViewResultsTabContent)

        self.selectFolderButton.clicked.connect(lambda: self._select_folder())

        self.selectedFolderText.textChanged.connect(self.directory_updated)
        self.viewButton.clicked.connect(self.view_maps)
        self.invertSelectionButton.clicked.connect(self.invert_selection)
        self.deselectAllButton.clicked.connect(self.deselect_all)
        self.initialSliceChooser.valueChanged.connect(self._shared_state.set_slice_index)
        self.initialDimensionChooser.valueChanged.connect(self._shared_state.set_dimension_index)
        self.initialSliceChooser.setMaximum(0)

    def open_dir(self, directory):
        self.selectedFolderText.setText(directory)
        self.directory_updated(directory)

    def _select_folder(self):
        initial_dir = self._shared_state.base_dir
        if self.selectedFolderText.text() != '':
            initial_dir = self.selectedFolderText.text()

        folder = QFileDialog().getExistingDirectory(caption='Select directory to view', directory=initial_dir)

        if os.path.isdir(folder):
            self.selectedFolderText.setText(folder)
            self._shared_state.base_dir = folder

    @pyqtSlot(str)
    def directory_updated(self, folder):
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)

        self._folder = folder
        result_files = glob.glob(os.path.join(folder, '*.nii*'))

        def get_name(img_path):
            return split_image_path(os.path.basename(img_path))[1]

        self._parameter_files = {get_name(f): get_name(f) for f in result_files}

        items_list = sorted(self._parameter_files.keys())
        selected_items = results_preselection_names(sorted(self._parameter_files.keys()))

        self.selectMaps.clear()
        self.selectMaps.addItems(items_list)

        for item in [self.selectMaps.item(index) for index in range(self.selectMaps.count())]:
            if item.text() in selected_items:
                item.setSelected(True)

        if items_list:
            shape = load_nifti(result_files[0]).shape
            maximum = shape[self.initialDimensionChooser.value()]
            self.initialSliceChooser.setMaximum(maximum)

            if self.initialSliceChooser.value() == 0 or self.initialSliceChooser.value() >= maximum:
                self.initialSliceChooser.setValue(maximum // 2.0)

            self.maximumIndexLabel.setText(str(maximum))

    def invert_selection(self):
        for item in [self.selectMaps.item(index) for index in range(self.selectMaps.count())]:
            item.setSelected(not item.isSelected())

    def deselect_all(self):
        for item in [self.selectMaps.item(index) for index in range(self.selectMaps.count())]:
            item.setSelected(False)

    def view_maps(self):
        maps_to_show = []
        for item in [self.selectMaps.item(index) for index in range(self.selectMaps.count())]:
            if item.isSelected():
                maps_to_show.append(item.text())

        data = SimpleDataInfo.from_dir(self._folder)

        config = MapPlotConfig()
        config.maps_to_show = maps_to_show
        config.dimension = self.initialDimensionChooser.value()
        config.slice_index = self.initialSliceChooser.value()

        start_gui(data=data, config=config, app_exec=False)

    def tab_opened(self):
        if self._shared_state.output_folder != '':
            self.selectedFolderText.setText(self._shared_state.output_folder)
