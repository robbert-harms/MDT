# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scientific_number_scroller_widget.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ScientificScroller(object):
    def setupUi(self, ScientificScroller):
        ScientificScroller.setObjectName("ScientificScroller")
        ScientificScroller.resize(283, 61)
        self.horizontalLayout = QtWidgets.QHBoxLayout(ScientificScroller)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mantissa = QDoubleSpinBoxDotSeparator(ScientificScroller)
        self.mantissa.setSpecialValueText("")
        self.mantissa.setDecimals(4)
        self.mantissa.setMinimum(-1000.0)
        self.mantissa.setMaximum(1000.0)
        self.mantissa.setSingleStep(0.01)
        self.mantissa.setObjectName("mantissa")
        self.horizontalLayout.addWidget(self.mantissa)
        self.label = QtWidgets.QLabel(ScientificScroller)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.exponent = QtWidgets.QSpinBox(ScientificScroller)
        self.exponent.setMinimum(-99)
        self.exponent.setObjectName("exponent")
        self.horizontalLayout.addWidget(self.exponent)
        self.horizontalLayout.setStretch(0, 1)

        self.retranslateUi(ScientificScroller)
        QtCore.QMetaObject.connectSlotsByName(ScientificScroller)

    def retranslateUi(self, ScientificScroller):
        _translate = QtCore.QCoreApplication.translate
        ScientificScroller.setWindowTitle(_translate("ScientificScroller", "Form"))
        self.label.setText(_translate("ScientificScroller", "E"))

from mdt.gui.widgets.decorator_widgets import QDoubleSpinBoxDotSeparator
