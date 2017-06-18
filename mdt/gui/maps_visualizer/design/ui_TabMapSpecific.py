# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TabMapSpecific.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TabMapSpecific(object):
    def setupUi(self, TabMapSpecific):
        TabMapSpecific.setObjectName("TabMapSpecific")
        TabMapSpecific.resize(445, 534)
        self.gridLayout = QtWidgets.QGridLayout(TabMapSpecific)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea_2 = QtWidgets.QScrollArea(TabMapSpecific)
        self.scrollArea_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 443, 532))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.selectedMap = QtWidgets.QComboBox(self.scrollAreaWidgetContents_2)
        self.selectedMap.setObjectName("selectedMap")
        self.verticalLayout.addWidget(self.selectedMap)
        self.frame = QtWidgets.QFrame(self.scrollAreaWidgetContents_2)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.mapSpecificOptionsPosition = QtWidgets.QGridLayout()
        self.mapSpecificOptionsPosition.setSpacing(0)
        self.mapSpecificOptionsPosition.setObjectName("mapSpecificOptionsPosition")
        self.gridLayout_4.addLayout(self.mapSpecificOptionsPosition, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout.addWidget(self.scrollArea_2, 0, 0, 1, 1)

        self.retranslateUi(TabMapSpecific)
        QtCore.QMetaObject.connectSlotsByName(TabMapSpecific)
        TabMapSpecific.setTabOrder(self.scrollArea_2, self.selectedMap)

    def retranslateUi(self, TabMapSpecific):
        _translate = QtCore.QCoreApplication.translate
        TabMapSpecific.setWindowTitle(_translate("TabMapSpecific", "Form"))

