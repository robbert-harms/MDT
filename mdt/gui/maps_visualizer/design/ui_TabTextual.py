# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TabTextual.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TabTextual(object):
    def setupUi(self, TabTextual):
        TabTextual.setObjectName("TabTextual")
        TabTextual.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(TabTextual)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(0)
        self.gridLayout.setVerticalSpacing(4)
        self.gridLayout.setObjectName("gridLayout")
        self.textConfigEdit = TextConfigEditor(TabTextual)
        self.textConfigEdit.setObjectName("textConfigEdit")
        self.gridLayout.addWidget(self.textConfigEdit, 0, 0, 1, 1)
        self.correctness_label = QtWidgets.QLabel(TabTextual)
        self.correctness_label.setAlignment(QtCore.Qt.AlignCenter)
        self.correctness_label.setWordWrap(True)
        self.correctness_label.setObjectName("correctness_label")
        self.gridLayout.addWidget(self.correctness_label, 1, 0, 1, 1)

        self.retranslateUi(TabTextual)
        QtCore.QMetaObject.connectSlotsByName(TabTextual)

    def retranslateUi(self, TabTextual):
        _translate = QtCore.QCoreApplication.translate
        TabTextual.setWindowTitle(_translate("TabTextual", "Form"))
        self.correctness_label.setText(_translate("TabTextual", "TextLabel"))

from ..widgets import TextConfigEditor
