# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_get_example_data.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GetExampleDataDialog(object):
    def setupUi(self, GetExampleDataDialog):
        GetExampleDataDialog.setObjectName("GetExampleDataDialog")
        GetExampleDataDialog.resize(691, 208)
        self.verticalLayout = QtWidgets.QVBoxLayout(GetExampleDataDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(GetExampleDataDialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(GetExampleDataDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.line = QtWidgets.QFrame(GetExampleDataDialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.label = QtWidgets.QLabel(GetExampleDataDialog)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtWidgets.QLabel(GetExampleDataDialog)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.outputFile = QtWidgets.QLineEdit(GetExampleDataDialog)
        self.outputFile.setObjectName("outputFile")
        self.horizontalLayout_5.addWidget(self.outputFile)
        self.outputFileSelect = QtWidgets.QPushButton(GetExampleDataDialog)
        self.outputFileSelect.setObjectName("outputFileSelect")
        self.horizontalLayout_5.addWidget(self.outputFileSelect)
        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.line_3 = QtWidgets.QFrame(GetExampleDataDialog)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(GetExampleDataDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(GetExampleDataDialog)
        self.buttonBox.accepted.connect(GetExampleDataDialog.accept)
        self.buttonBox.rejected.connect(GetExampleDataDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(GetExampleDataDialog)
        GetExampleDataDialog.setTabOrder(self.outputFile, self.outputFileSelect)

    def retranslateUi(self, GetExampleDataDialog):
        _translate = QtCore.QCoreApplication.translate
        GetExampleDataDialog.setWindowTitle(_translate("GetExampleDataDialog", "Get example data"))
        self.label_3.setText(_translate("GetExampleDataDialog", "Get example data"))
        self.label_4.setText(_translate("GetExampleDataDialog", "Loads the MDT example data from the installation files"))
        self.label.setText(_translate("GetExampleDataDialog", "This will write the MDT example data (b1k_b2k and b6k datasets) to the indicated directory. You can use this data for testing MDT on your computer. These example datasets are contained within the MDT package and as such are available on every machine with MDT installed."))
        self.label_7.setText(_translate("GetExampleDataDialog", "Output folder:"))
        self.outputFileSelect.setText(_translate("GetExampleDataDialog", "File browser"))

