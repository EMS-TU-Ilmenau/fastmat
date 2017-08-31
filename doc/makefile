#
#  doc/makefile
# -------------------------------------------------- part of the fastmat package
#
#  makefile for building documentation of fastMat
#
#
#  Author      : wcw
#  Introduced  : 2016-07-06
#------------------------------------------------------------------------------
#
#  Copyright 2016 Sebastian Semper, Christoph Wagner
#      https://www.tu-ilmenau.de/ems/
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#------------------------------------------------------------------------------
.PHONY: default
default: doc

################################################################################
###  PROJECT STRUCTURE
################################################################################
#
#  * .......................... Project root
#  |
#  +--o> fastmat .............. Codebase directory: $(CODE)
#  |  |                         This directory contains class source code.
#  |  |
#  |  +--o> algs .............. Algorithm directory: $(CODE_ALGS)
#  |                            This directory contains algorithm source code.
#  |
#  +--o> doc .................. base directory for documentation framework
#  |  |
#  |  +> makefile ............. this file
#  |  |                         invoke make in this directory to run automatic
#  |  |                         documentation generation. Filenames and paths
#  |  |                         e.g. $(CODEBASE) must be specified relative to
#  |  |                         the doc basepath
#  |  |
#  |  +--o> $(RESULTS) ........ output directory
#  |                            Here all benchmark output will be stored.
#  |                            NOTE: This directory will be completely removed
#  |                            by 'make cleanall'!
#  |
#  +--+> util ................. directory for helper and utility scripts
#     |
#     +---> $(SCRIPT_BEE) ..... script for python module-function interface
#                               Collects and presents information about the in-
#                               ternals of the package. Use this script to re-
#                               trieve string output for certain tasks (i.e. do-
#                               cumentation, unit testing, benchmarking or
#                               structural queries of the package)
#
#
################################################################################
###  PLATFORM SPECIFIC DEFINITIONS
################################################################################

ifeq ($(OS),Windows_NT)
ECHO=@echo $(1)
RM=del /F
RMR=deltree
MKDIR=mkdir
PSEP=$(strip \)
else
ECHO=@echo "$(1)"
RM=rm -f
RMR=rm -rf
MKDIR=mkdir -p
PSEP=/
endif


################################################################################
###  LOCAL VARIABLES (user-defined in case needed)
################################################################################

# directory names of the fastmat codebase
NAME=fastmat

# name of the directories in the doc section
RESULTS=results
OUTPUT=output
TEX=tex

# name of the output files
NAME_DOCU=fastmatDoc
CAL_DATA=$(RESULTS)$(PSEP)calibration.json

# name of the tex resource directories
TEX_TRAFOS=$(OUTPUT)$(PSEP)trafos.tex
TEX_ALGS=$(OUTPUT)$(PSEP)algs.tex
TEXRESOURCES:=$(TEX_TRAFOS) $(TEX_ALGS)\
	$(foreach file,funcs intro head,$(TEX)$(PSEP)$(file).tex)

# links to generator scripts
SCRIPT=..$(PSEP)util
SCRIPT_BEE=$(SCRIPT)$(PSEP)bee.py
SCRIPT_RESOURCES=$(SCRIPT_BEE) makefile

# python version to use for doc generation (may be overwritten on command line)
PYTHON=python

# shortcut to scripts
BEE=$(PYTHON) $(SCRIPT_BEE)


################################################################################
###  LOCAL VARIABLES (generated, NO USER CHANGES BEYOUND THIS POINT)
################################################################################

# states the file extensions for different file types
EXT_RESULT=csv
EXT_TEX_OUTPUT=pdf aux log toc bbl lof lot blg out nav snm


################################################################################
###  DEFINES AND MACROS (naming functions)
################################################################################

# NAME_TEX_OUTPUT_FILES
#   Return all intermediate and temporary files output by pdflatex for $(1).
#   EXAMPLE: $(call ~~~,fastmatDoc)
#       ==> 'fastmatDoc.pdf fastmatDoc.aux fastmatDoc.log [...]'
NAME_TEX_OUTPUT_FILES=$(foreach ext,$(EXT_TEX_OUTPUT),$(1).$(ext))


################################################## File-list generation

# holds five lines containing the output of the util/bee commands used in this
# makefile. BEE_LINE is a call makro to dispatch the dump output.
LIST_BEEDUMP:=$(shell $(BEE) list makedump)

BEE_LINE=$(subst :, ,$(word $(1),$(subst ;, ,$(LIST_BEEDUMP))))

LIST_CLASSES:=$(call BEE_LINE, 1)
LIST_ALGS:=$(call BEE_LINE, 2)


################################################################################
###  BUILD TARGETS
################################################################################

# target 'compile': Compile documentation
.PHONY: compile
compile: | $(TEX_TRAFOS) $(TEX_ALGS) $(CAL_DATA) $(NAME_DOCU).pdf


# target 'doc': Build documentation including all benchmarks if needed
.PHONY: doc
doc: | compile

# target 'benchmarks': Perform all benchmarks
.PHONY: benchmarks
benchmarks: | $(CAL_DATA)
	@$(BEE) benchmark -p $(RESULTS) -o . $(OPTIONS)


# target 'clean':
.PHONY: clean
clean:
	@$(call ECHO, ***  Cleaning up docu files from last build.)
	@-$(RM) $(call NAME_TEX_OUTPUT_FILES,$(NAME_DOCU))
	@-$(RMR) $(OUTPUT)
	@$(call ECHO, ***)


# target 'cleanall':
.PHONY: cleanall
cleanall: | clean
	@$(call ECHO, ***  Cleaning up performance data.)
	@-$(RMR) $(RESULTS)
	@$(call ECHO, ***)


# target 'rebuild':
.PHONY: rebuild
rebuild: | cleanall all


################################################################################
###  INTERNAL BUILD TARGETS
################################################################################

# this line provokes execution of '.welcome' regardless of the
# target make was called with. After completion make restarts
# for regular execution
-include .welcome

# target '.welcome': Print some internal variables and lists
.PHONY: .welcome
.welcome:
	@$(call ECHO, *******************************************)
	@$(call ECHO, *  MAKE - building fastMat documentation  *)
	@$(call ECHO, *******************************************)
	@$(call ECHO, ***)


# target '.debug': Print extended set of variables and lists for debug
.PHONY: .debug
.debug:
	@$(call ECHO, ***  Extended debug output)
	@$(call ECHO, *** -----------------------)
	@$(call ECHO, *** units containing classes:)
	@$(call ECHO,$(LIST_CLASSES))
	@$(call ECHO, *** units containing algorithms:)
	@$(call ECHO,$(LIST_ALGS))
	@$(call ECHO,.)

################################################################################
###  RESOURCE TARGETS
################################################################################

################################################## Calibration data
$(CAL_DATA): $(SCRIPT_RESOURCES)
	@$(call ECHO, ***  Generating calibration data )
	@$(BEE) calibrate all -o $(CAL_DATA)


################################################## Compilation of class docu
$(TEX_TRAFOS): $(SCRIPT_RESOURCES) $(CAL_DATA)
	@$(call ECHO, ***  Extracting documentation from source code )
	@$(call ECHO, ***   > modules: $(LIST_CLASSES))
	@$(call ECHO, ***   > output: $@)
	@$(MKDIR) $(OUTPUT)
	@-$(RM) $@
	@$(BEE) documentation $(LIST_CLASSES) $(OPTIONS)\
		-p $(RESULTS) -o $@ -c $(CAL_DATA)


################################################## Compilation of algorithm docu
$(TEX_ALGS): $(SCRIPT_RESOURCES) $(CAL_DATA)
	@$(call ECHO, ***  Extracting documentation from source code )
	@$(call ECHO, ***   > modules: $(LIST_ALGS))
	@$(call ECHO, ***   > output: $@)
	@$(MKDIR) $(OUTPUT)
	@-$(RM) $@
	@$(BEE) documentation $(LIST_ALGS) $(OPTIONS)\
		-p $(RESULTS) -o $@ -c $(CAL_DATA)

################################################## LaTeX Compilation targets
# these suffixes will be used in suffix recipes
#.SUFFIXES: .bib .toc .aux .bbl


# target '%.toc': Build .toc files
%.toc: $(TEXRESOURCES) $(SCRIPT_RESOURCES) %.tex
	pdflatex -shell-escape -draftmode -interaction=nonstopmode $*


# target '%.aux': Build .aux files
%.aux: $(TEXRESOURCES) $(SCRIPT_RESOURCES) %.tex %.toc
	pdflatex -shell-escape -draftmode -interaction=nonstopmode $*
	$(RM) $*.pdf


# target '%.bbl': Build .bbl files
%.bbl: %.aux
ifneq ($(wildcard $*.bib), )
	bibtex $(basename $*)
else
	@$(call ECHO, ***  No bibliography found. Skipping bibtex.)
endif

################################################## Additional dependencies

# target '$(NAME_DOCU).pdf': Build docu pdf file with additional targets
$(NAME_DOCU).pdf: $(TEXRESOURCES) $(SCRIPT_RESOURCES) $(TEX_TRAFOS) $(TEX_ALGS)\
	$(NAME_DOCU).tex $(NAME_DOCU).aux
	@$(call ECHO, ***  Prepared all depencies for $@, compiling doc ...)
	pdflatex -shell-escape -interaction=nonstopmode $(NAME_DOCU)
