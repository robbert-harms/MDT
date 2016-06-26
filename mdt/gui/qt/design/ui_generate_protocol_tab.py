# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'generate_protocol_tab.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GenerateProtocolTabContent(object):
    def setupUi(self, GenerateProtocolTabContent):
        GenerateProtocolTabContent.setObjectName("GenerateProtocolTabContent")
        GenerateProtocolTabContent.setWindowModality(QtCore.Qt.NonModal)
        GenerateProtocolTabContent.resize(400, 302)
        self.verticalLayout = QtWidgets.QVBoxLayout(GenerateProtocolTabContent)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(GenerateProtocolTabContent)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(GenerateProtocolTabContent)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)

        self.retranslateUi(GenerateProtocolTabContent)
        QtCore.QMetaObject.connectSlotsByName(GenerateProtocolTabContent)

    def retranslateUi(self, GenerateProtocolTabContent):
        _translate = QtCore.QCoreApplication.translate
        GenerateProtocolTabContent.setWindowTitle(_translate("GenerateProtocolTabContent", "Form"))
        self.label.setText(_translate("GenerateProtocolTabContent", "TextLabel"))
        self.label_2.setText(_translate("GenerateProtocolTabContent", "TextLabel"))

