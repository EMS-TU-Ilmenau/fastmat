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

# name of the tex resource directories
TEX_TRAFOS=$(OUTPUT)$(PSEP)trafos.tex
TEX_ALGS=$(OUTPUT)$(PSEP)algs.tex
TEXRESOURCES:=$(TEX_TRAFOS) $(TEX_ALGS)\
	$(foreach file,funcs intro head,$(TEX)$(PSEP)$(file).tex)

# links to generator scripts
SCRIPT=..$(PSEP)util
SCRIPT_BEE=$(SCRIPT)$(PSEP)bee.py
SCRIPT_RESOURCES=$(foreach file,benchmark.py documentation.py printing.py\
	statistics.py test.py, $(SCRIPT)$(PSEP)routines$(PSEP)$(file))\
	$(SCRIPT_BEE) makefile

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


# NAME_RESULT_FROM_BENCHMARK
#   Return the benchmark result filename for a given unit benchmark target
#   EXAMPLE: $(call ~~~,'algs.ISTA.performance')
#       ==> 'results/algs.ISTA.performance.csv'
NAME_RESULT_FROM_BENCHMARK=$(RESULTS)$(PSEP)$(1).$(EXT_RESULT)

################################################## File-list generation

# holds five lines containing the output of the util/bee commands used in this
# makefile. BEE_LINE is a call makro to dispatch the dump output.
LIST_BEEDUMP:=$(shell $(BEE) list makedump)

BEE_LINE=$(subst :, ,$(word $(1),$(subst ;, ,$(LIST_BEEDUMP))))

LIST_CLASSES:=$(call BEE_LINE, 1)
LIST_CLASSES_SOURCE:=$(call BEE_LINE, 2)
LIST_ALGS:=$(call BEE_LINE, 3)
LIST_ALGS_SOURCE:=$(call BEE_LINE, 4)
LIST_BENCHMARK_TARGETS:=$(call BEE_LINE, 5)
LIST_BENCHMARK_RESULTS:=$(foreach target,$(LIST_BENCHMARK_TARGETS),$(call\
	NAME_RESULT_FROM_BENCHMARK,$(target)))

LIST_SOURCES:=$(LIST_CLASSES_SOURCE) $(LIST_ALGS_SOURCE)

################################################################################
###  BUILD TARGETS
################################################################################

# target 'compile': Only compile documentation, do not do benchmarks
.PHONY: compile
compile: | $(NAME_DOCU).pdf


# target 'doc': Build documentation including all benchmarks
.PHONY: doc
# conditional: if no benchmarks exist yet, build them all in one run. Otherwise
#              decide and run each target
ifeq ($(wildcard $(RESULTS)$(PSEP)*.$(EXT_RESULT)),)
doc: | allbenchmarks compile
else
doc: | benchmarks compile
endif


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
rebuild: | cleanall allbenchmarks all


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
	@$(call ECHO, *** units containing classes (and their source files):)
	@$(call ECHO,$(LIST_CLASSES))
	@$(call ECHO,$(LIST_CLASSES_SOURCE))
	@$(call ECHO, *** units containing algorithms (and their source files):)
	@$(call ECHO,$(LIST_ALGS))
	@$(call ECHO,$(LIST_ALGS_SOURCE))
	@$(call ECHO, *** available benchmark targets (and their result files):)
	@$(call ECHO,$(LIST_BENCHMARK_TARGETS))
	@$(call ECHO,$(LIST_BENCHMARK_RESULTS))
	@$(call ECHO,.)

################################################################################
###  RESOURCE TARGETS
################################################################################

################################################## Compilation of class docu
$(TEX_TRAFOS): $(SCRIPT_RESOURCES) $(LIST_CLASSES_SOURCE)
	@$(call ECHO, ***  Extracting documentation from source code )
	@$(call ECHO, ***   > modules: $(LIST_CLASSES))
	@$(call ECHO, ***   > output: $@)
	@$(MKDIR) $(OUTPUT)
	@-$(RM) $@
	@$(BEE) documentation $(LIST_CLASSES) -p $(RESULTS) > $@


################################################## Compilation of algorithm docu
$(TEX_ALGS): $(SCRIPT_RESOURCES) $(LIST_ALGS_SOURCE)
	@$(call ECHO, ***  Extracting documentation from source code )
	@$(call ECHO, ***   > modules: $(LIST_ALGS))
	@$(call ECHO, ***   > output: $@)
	@$(MKDIR) $(OUTPUT)
	@-$(RM) $@
	@$(BEE) documentation $(LIST_ALGS) -p $(RESULTS) > $@

################################################## Evaluation of performance

# specify an individual target for each benchmark
define BENCHMARK_rule
$(2): $$(SCRIPT_RESOURCES)
	@$(call ECHO, ***  Running benchmark for target '$(1)' [$(2)])
	@$(BEE) benchmark -s $(1) -p $$(RESULTS) $$(OPTIONS)
endef
$(foreach target,$(LIST_BENCHMARK_TARGETS),$(eval $(call\
	BENCHMARK_rule,$(target),$(call NAME_RESULT_FROM_BENCHMARK,$(target)))))


# target 'benchmarks': Run benchmarks one-by-one. Regeneration only when needed.
.PHONY: benchmarks
benchmarks: $(LIST_BENCHMARK_RESULTS)


# target 'allbenchmarks': Run benchmarks in one run: faster, but reruns existing
.PHONY: allbenchmarks
allbenchmarks: $(SCRIPT_RESOURCES) $(LIST_SOURCES)
	@$(call ECHO, ***  Running benchmarks for all targets)
	@$(BEE) benchmark -p $(RESULTS) $(OPTIONS)

################################################## LaTeX Compilation targets

# these suffixes will be used in suffix recipes
#.SUFFIXES: .bib .toc .aux .bbl


# target '%.toc': Build .toc files
%.toc: %.tex
	pdflatex -shell-escape -draftmode -interaction=nonstopmode $*


# target '%.aux': Build .aux files
%.aux: %.tex %.toc
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
$(NAME_DOCU).pdf: $(TEXRESOURCES) $(SCRIPT_RESOURCES) $(LIST_SOURCES) \
	$(NAME_DOCU).tex $(NAME_DOCU).aux
	@$(call ECHO, ***  Prepared all depencies for $@, compiling doc ...)
	pdflatex --shell-escape -interaction=nonstopmode $(NAME_DOCU)
