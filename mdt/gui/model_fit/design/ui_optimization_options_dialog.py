# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'optimization_options_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_OptimizationOptionsDialog:
    def setupUi(self, OptimizationOptionsDialog):
        OptimizationOptionsDialog.setObjectName("OptimizationOptionsDialog")
        OptimizationOptionsDialog.resize(843, 337)
        self.verticalLayout = QtWidgets.QVBoxLayout(OptimizationOptionsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.line = QtWidgets.QFrame(OptimizationOptionsDialog)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_12 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 4, 2, 1, 1)
        self.label = QtWidgets.QLabel(OptimizationOptionsDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.defaultOptimizer_True = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.defaultOptimizer_True.setChecked(True)
        self.defaultOptimizer_True.setObjectName("defaultOptimizer_True")
        self.defaultOptimizerGroup = QtWidgets.QButtonGroup(OptimizationOptionsDialog)
        self.defaultOptimizerGroup.setObjectName("defaultOptimizerGroup")
        self.defaultOptimizerGroup.addButton(self.defaultOptimizer_True)
        self.horizontalLayout.addWidget(self.defaultOptimizer_True)
        self.defaultOptimizer_False = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.defaultOptimizer_False.setObjectName("defaultOptimizer_False")
        self.defaultOptimizerGroup.addButton(self.defaultOptimizer_False)
        self.horizontalLayout.addWidget(self.defaultOptimizer_False)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.floatPrecision = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.floatPrecision.setChecked(True)
        self.floatPrecision.setObjectName("floatPrecision")
        self.floatPrecisionGroup = QtWidgets.QButtonGroup(OptimizationOptionsDialog)
        self.floatPrecisionGroup.setObjectName("floatPrecisionGroup")
        self.floatPrecisionGroup.addButton(self.floatPrecision)
        self.horizontalLayout_2.addWidget(self.floatPrecision)
        self.doublePrecision = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.doublePrecision.setObjectName("doublePrecision")
        self.floatPrecisionGroup.addButton(self.doublePrecision)
        self.horizontalLayout_2.addWidget(self.doublePrecision)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 7, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.recalculateAll_True = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.recalculateAll_True.setObjectName("recalculateAll_True")
        self.recalculateAllGroup = QtWidgets.QButtonGroup(OptimizationOptionsDialog)
        self.recalculateAllGroup.setObjectName("recalculateAllGroup")
        self.recalculateAllGroup.addButton(self.recalculateAll_True)
        self.horizontalLayout_3.addWidget(self.recalculateAll_True)
        self.recalculateAll_False = QtWidgets.QRadioButton(OptimizationOptionsDialog)
        self.recalculateAll_False.setChecked(True)
        self.recalculateAll_False.setObjectName("recalculateAll_False")
        self.recalculateAllGroup.addButton(self.recalculateAll_False)
        self.horizontalLayout_3.addWidget(self.recalculateAll_False)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_3, 7, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 0, 2, 1, 1)
        self.patience = QtWidgets.QLineEdit(OptimizationOptionsDialog)
        self.patience.setEnabled(False)
        self.patience.setObjectName("patience")
        self.gridLayout.addWidget(self.patience, 4, 1, 1, 1)
        self.line_4 = QtWidgets.QFrame(OptimizationOptionsDialog)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 6, 0, 1, 3)
        self.label_11 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 3, 2, 1, 1)
        self.optimizationRoutine = QtWidgets.QComboBox(OptimizationOptionsDialog)
        self.optimizationRoutine.setEnabled(False)
        self.optimizationRoutine.setEditable(False)
        self.optimizationRoutine.setObjectName("optimizationRoutine")
        self.gridLayout.addWidget(self.optimizationRoutine, 3, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(OptimizationOptionsDialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(OptimizationOptionsDialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(OptimizationOptionsDialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(OptimizationOptionsDialog)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(OptimizationOptionsDialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)
        self.line_5 = QtWidgets.QFrame(OptimizationOptionsDialog)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 1, 0, 1, 3)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem3)
        self.line_3 = QtWidgets.QFrame(OptimizationOptionsDialog)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout.addWidget(self.line_3)
        self.buttonBox = QtWidgets.QDialogButtonBox(OptimizationOptionsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(OptimizationOptionsDialog)
        self.buttonBox.accepted.connect(OptimizationOptionsDialog.accept)
        self.buttonBox.rejected.connect(OptimizationOptionsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(OptimizationOptionsDialog)
        OptimizationOptionsDialog.setTabOrder(self.floatPrecision, self.doublePrecision)
        OptimizationOptionsDialog.setTabOrder(self.doublePrecision, self.defaultOptimizer_True)
        OptimizationOptionsDialog.setTabOrder(self.defaultOptimizer_True, self.defaultOptimizer_False)
        OptimizationOptionsDialog.setTabOrder(self.defaultOptimizer_False, self.optimizationRoutine)
        OptimizationOptionsDialog.setTabOrder(self.optimizationRoutine, self.patience)
        OptimizationOptionsDialog.setTabOrder(self.patience, self.recalculateAll_True)
        OptimizationOptionsDialog.setTabOrder(self.recalculateAll_True, self.recalculateAll_False)

    def retranslateUi(self, OptimizationOptionsDialog):
        _translate = QtCore.QCoreApplication.translate
        OptimizationOptionsDialog.setWindowTitle(_translate("OptimizationOptionsDialog", "Optimization options"))
        self.label_3.setText(_translate("OptimizationOptionsDialog", "Optimization options"))
        self.label_4.setText(_translate("OptimizationOptionsDialog", "Advanced options for the model fitting procedure"))
        self.label_12.setText(_translate("OptimizationOptionsDialog", "(Scales the number of iterations)"))
        self.label.setText(_translate("OptimizationOptionsDialog", "Recalculate all:"))
        self.defaultOptimizer_True.setText(_translate("OptimizationOptionsDialog", "Yes   "))
        self.defaultOptimizer_False.setText(_translate("OptimizationOptionsDialog", "No"))
        self.floatPrecision.setText(_translate("OptimizationOptionsDialog", "Float"))
        self.doublePrecision.setText(_translate("OptimizationOptionsDialog", "Double"))
        self.label_13.setText(_translate("OptimizationOptionsDialog", "(For cascades, if we want to recalculate the entire chain)"))
        self.recalculateAll_True.setText(_translate("OptimizationOptionsDialog", "Yes   "))
        self.recalculateAll_False.setText(_translate("OptimizationOptionsDialog", "No"))
        self.label_15.setText(_translate("OptimizationOptionsDialog", "(The precision for the calculations)"))
        self.label_11.setText(_translate("OptimizationOptionsDialog", "(Manual select the routine to use)"))
        self.label_2.setText(_translate("OptimizationOptionsDialog", "Use default optimizer:"))
        self.label_5.setText(_translate("OptimizationOptionsDialog", "Optimization routine:"))
        self.label_6.setText(_translate("OptimizationOptionsDialog", "Patience:"))
        self.label_8.setText(_translate("OptimizationOptionsDialog", "Float precision:"))
        self.label_10.setText(_translate("OptimizationOptionsDialog", "(Enables manual selection of the optimization routine)"))

