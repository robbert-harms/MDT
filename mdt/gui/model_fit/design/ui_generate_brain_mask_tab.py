# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'generate_brain_mask_tab.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GenerateBrainMaskTabContent:
    def setupUi(self, GenerateBrainMaskTabContent):
        GenerateBrainMaskTabContent.setObjectName("GenerateBrainMaskTabContent")
        GenerateBrainMaskTabContent.resize(827, 427)
        self.verticalLayout = QtWidgets.QVBoxLayout(GenerateBrainMaskTabContent)
        self.verticalLayout.setContentsMargins(-1, 11, -1, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.line = QtWidgets.QFrame(GenerateBrainMaskTabContent)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setLineWidth(1)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 5, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.selectImageButton = QtWidgets.QPushButton(GenerateBrainMaskTabContent)
        self.selectImageButton.setObjectName("selectImageButton")
        self.horizontalLayout_2.addWidget(self.selectImageButton)
        self.selectedImageText = QtWidgets.QLineEdit(GenerateBrainMaskTabContent)
        self.selectedImageText.setText("")
        self.selectedImageText.setObjectName("selectedImageText")
        self.horizontalLayout_2.addWidget(self.selectedImageText)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(0, -1, 0, -1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.medianRadiusInput = QtWidgets.QSpinBox(GenerateBrainMaskTabContent)
        self.medianRadiusInput.setMinimum(1)
        self.medianRadiusInput.setProperty("value", 4)
        self.medianRadiusInput.setObjectName("medianRadiusInput")
        self.horizontalLayout_4.addWidget(self.medianRadiusInput)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_4, 4, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.selectOutputButton = QtWidgets.QPushButton(GenerateBrainMaskTabContent)
        self.selectOutputButton.setObjectName("selectOutputButton")
        self.horizontalLayout_3.addWidget(self.selectOutputButton)
        self.selectedOutputText = QtWidgets.QLineEdit(GenerateBrainMaskTabContent)
        self.selectedOutputText.setObjectName("selectedOutputText")
        self.horizontalLayout_3.addWidget(self.selectedOutputText)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 2, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 4, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.selectProtocolButton = QtWidgets.QPushButton(GenerateBrainMaskTabContent)
        self.selectProtocolButton.setObjectName("selectProtocolButton")
        self.horizontalLayout_5.addWidget(self.selectProtocolButton)
        self.selectedProtocolText = QtWidgets.QLineEdit(GenerateBrainMaskTabContent)
        self.selectedProtocolText.setObjectName("selectedProtocolText")
        self.horizontalLayout_5.addWidget(self.selectedProtocolText)
        self.gridLayout.addLayout(self.horizontalLayout_5, 1, 1, 1, 1)
        self.line_3 = QtWidgets.QFrame(GenerateBrainMaskTabContent)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 3, 0, 1, 3)
        self.label_10 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_3.setMinimumSize(QtCore.QSize(0, 0))
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 6, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.numberOfPassesInput = QtWidgets.QSpinBox(GenerateBrainMaskTabContent)
        self.numberOfPassesInput.setMinimum(1)
        self.numberOfPassesInput.setProperty("value", 4)
        self.numberOfPassesInput.setObjectName("numberOfPassesInput")
        self.horizontalLayout_6.addWidget(self.numberOfPassesInput)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_6, 5, 1, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.finalThresholdInput = QtWidgets.QDoubleSpinBox(GenerateBrainMaskTabContent)
        self.finalThresholdInput.setLocale(QtCore.QLocale(QtCore.QLocale.C, QtCore.QLocale.AnyCountry))
        self.finalThresholdInput.setPrefix("")
        self.finalThresholdInput.setSuffix("")
        self.finalThresholdInput.setMaximum(1000000.0)
        self.finalThresholdInput.setObjectName("finalThresholdInput")
        self.horizontalLayout_7.addWidget(self.finalThresholdInput)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_7, 6, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 4, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 5, 2, 1, 1)
        self.label_16 = QtWidgets.QLabel(GenerateBrainMaskTabContent)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 6, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.line_2 = QtWidgets.QFrame(GenerateBrainMaskTabContent)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 6, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.generateButton = QtWidgets.QPushButton(GenerateBrainMaskTabContent)
        self.generateButton.setEnabled(False)
        self.generateButton.setObjectName("generateButton")
        self.horizontalLayout.addWidget(self.generateButton)
        self.viewButton = QtWidgets.QPushButton(GenerateBrainMaskTabContent)
        self.viewButton.setEnabled(False)
        self.viewButton.setObjectName("viewButton")
        self.horizontalLayout.addWidget(self.viewButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)

        self.retranslateUi(GenerateBrainMaskTabContent)
        QtCore.QMetaObject.connectSlotsByName(GenerateBrainMaskTabContent)
        GenerateBrainMaskTabContent.setTabOrder(self.selectImageButton, self.selectedImageText)
        GenerateBrainMaskTabContent.setTabOrder(self.selectedImageText, self.selectProtocolButton)
        GenerateBrainMaskTabContent.setTabOrder(self.selectProtocolButton, self.selectedProtocolText)
        GenerateBrainMaskTabContent.setTabOrder(self.selectedProtocolText, self.selectOutputButton)
        GenerateBrainMaskTabContent.setTabOrder(self.selectOutputButton, self.selectedOutputText)
        GenerateBrainMaskTabContent.setTabOrder(self.selectedOutputText, self.medianRadiusInput)
        GenerateBrainMaskTabContent.setTabOrder(self.medianRadiusInput, self.numberOfPassesInput)
        GenerateBrainMaskTabContent.setTabOrder(self.numberOfPassesInput, self.finalThresholdInput)
        GenerateBrainMaskTabContent.setTabOrder(self.finalThresholdInput, self.generateButton)
        GenerateBrainMaskTabContent.setTabOrder(self.generateButton, self.viewButton)

    def retranslateUi(self, GenerateBrainMaskTabContent):
        _translate = QtCore.QCoreApplication.translate
        GenerateBrainMaskTabContent.setWindowTitle(_translate("GenerateBrainMaskTabContent", "Form"))
        self.label.setText(_translate("GenerateBrainMaskTabContent", "Generate brian mask"))
        self.label_2.setText(_translate("GenerateBrainMaskTabContent", "Create a whole brain mask using the median-otsu algorithm."))
        self.label_5.setText(_translate("GenerateBrainMaskTabContent", "(To create one, please see the tab \"Generate protocol file\")"))
        self.label_12.setText(_translate("GenerateBrainMaskTabContent", "Number of passes:"))
        self.selectImageButton.setText(_translate("GenerateBrainMaskTabContent", "Browse"))
        self.label_4.setText(_translate("GenerateBrainMaskTabContent", "(Select the 4d diffusion weighted image)"))
        self.selectOutputButton.setText(_translate("GenerateBrainMaskTabContent", "Browse"))
        self.label_8.setText(_translate("GenerateBrainMaskTabContent", "Select output file:"))
        self.label_11.setText(_translate("GenerateBrainMaskTabContent", "Median radius:"))
        self.selectProtocolButton.setText(_translate("GenerateBrainMaskTabContent", "Browse"))
        self.label_10.setText(_translate("GenerateBrainMaskTabContent", "(Default is <volume_name>_mask.nii.gz)"))
        self.label_6.setText(_translate("GenerateBrainMaskTabContent", "Select 4d image:"))
        self.label_3.setText(_translate("GenerateBrainMaskTabContent", "Select protocol file:"))
        self.label_13.setText(_translate("GenerateBrainMaskTabContent", "Final threshold:"))
        self.label_14.setText(_translate("GenerateBrainMaskTabContent", "(Radius (in voxels) of the applied median filter)"))
        self.label_15.setText(_translate("GenerateBrainMaskTabContent", "(Number of median filter passes)"))
        self.label_16.setText(_translate("GenerateBrainMaskTabContent", "(Additional masking threshold as a signal intensity)"))
        self.generateButton.setText(_translate("GenerateBrainMaskTabContent", "Generate"))
        self.viewButton.setText(_translate("GenerateBrainMaskTabContent", "View mask"))

