###### EDIT ##################### 

# Directory with ui and resource files
QT_RESOURCE_DIR = ./
 
# Directory for compiled QT_RESOURCES
QT_COMPILED_DIR = ../../gui/qt/design
 
# UI files to compile
QT_UI_FILES = gui_single.ui

# Qt resource files to compile
QT_RESOURCES = gui_single.qrc
 
# the binaries
PYUIC = pyuic5
PYRCC = pyrcc5

 
#################################
# DO NOT EDIT FOLLOWING

help:
	@echo "clean - remove all build designer files"
	@echo "ui - build only thy UI files"
	@echo "resources - build only the resources"
	@echo "all - build both the UI and the resources"

QT_COMPILED_UI = $(QT_UI_FILES:%.ui=$(QT_COMPILED_DIR)/ui_%.py)
QT_COMPILED_RESOURCES = $(QT_RESOURCES:%.qrc=$(QT_COMPILED_DIR)/rc_%.py)
 
all : clean resources ui  

resources : $(QT_COMPILED_RESOURCES) 

ui : $(QT_COMPILED_UI)
 
$(QT_COMPILED_DIR)/ui_%.py : $(QT_RESOURCE_DIR)/%.ui
	$(PYUIC) $< -o $@
 
$(QT_COMPILED_DIR)/%_rc.py : $(QT_RESOURCE_DIR)/%.qrc
	$(PYRCC) $< -o $@
 
clean : 	
	$(RM) $(QT_COMPILED_UI) $(QT_COMPILED_RESOURCES) $(QT_COMPILED_UI:.py=.pyc) $(QT_COMPILED_RESOURCES:.py=.pyc)  
