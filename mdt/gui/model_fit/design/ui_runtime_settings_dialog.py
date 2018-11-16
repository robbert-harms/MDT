# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'runtime_settings_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_RuntimeSettingsDialog(object):
    def setupUi(self, RuntimeSettingsDialog):
        RuntimeSettingsDialog.setObjectName("RuntimeSettingsDialog")
        RuntimeSettingsDialog.resize(844, 243)
        self.verticalLayout = QtWidgets.QVBoxLayout(RuntimeSettingsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(RuntimeSettingsDialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(RuntimeSettingsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.line = QtWidgets.QFrame(RuntimeSettingsDialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.cldevicesSelection = QtWidgets.QListWidget(RuntimeSettingsDialog)
        self.cldevicesSelection.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.cldevicesSelection.setObjectName("cldevicesSelection")
        self.gridLayout.addWidget(self.cldevicesSelection, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(RuntimeSettingsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(RuntimeSettingsDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.line_3 = QtWidgets.QFrame(RuntimeSettingsDialog)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(RuntimeSettingsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(RuntimeSettingsDialog)
        self.buttonBox.accepted.connect(RuntimeSettingsDialog.accept)
        self.buttonBox.rejected.connect(RuntimeSettingsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(RuntimeSettingsDialog)

    def retranslateUi(self, RuntimeSettingsDialog):
        _translate = QtCore.QCoreApplication.translate
        RuntimeSettingsDialog.setWindowTitle(_translate("RuntimeSettingsDialog", "Runtime settings"))
        self.label_3.setText(_translate("RuntimeSettingsDialog", "Runtime settings"))
        self.label_4.setText(_translate("RuntimeSettingsDialog", "Runtime settings for all compute operations."))
        self.label_10.setText(_translate("RuntimeSettingsDialog", "(Select the devices you would like to use)"))
        self.label.setText(_translate("RuntimeSettingsDialog", "OpenCL devices:"))

