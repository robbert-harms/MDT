###### EDIT ##################### 

# Directory with ui and resource files
QT_RESOURCE_DIR = ./
 
# Directory for compiled QT_RESOURCES
QT_COMPILED_DIR = ../../../gui/widgets/design
 
# UI files to compile
QT_UI_FILES = $(shell ls $(QT_RESOURCE-DIR)*.ui)

# Qt resource files to compile
QT_RESOURCES = 
 
# the binaries
PYUIC = pyuic5 --from-imports
PYRCC = pyrcc5

 
#################################
# DO NOT EDIT FOLLOWING

all : clean resources ui  

help:
	@echo "clean - remove all build designer files"
	@echo "ui - build only thy UI files"
	@echo "resources - build only the resources"
	@echo "all - build both the UI and the resources"

QT_COMPILED_UI = $(QT_UI_FILES:%.ui=$(QT_COMPILED_DIR)/ui_%.py)
QT_COMPILED_RESOURCES = $(QT_RESOURCES:%.qrc=$(QT_COMPILED_DIR)/%_rc.py)

resources : $(QT_COMPILED_RESOURCES) 

ui : $(QT_COMPILED_UI)
 
$(QT_COMPILED_DIR)/ui_%.py : $(QT_RESOURCE_DIR)/%.ui
	$(PYUIC) $< -o $@
 
$(QT_COMPILED_DIR)/%_rc.py : $(QT_RESOURCE_DIR)/%.qrc
	$(PYRCC) $< -o $@
 
clean : 	
	$(RM) -f $(QT_COMPILED_DIR)/ui_*.py $(QT_COMPILED_DIR)/*_rc.py
