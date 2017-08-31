# -*- coding: utf-8 -*-
'''
  util/routines/printing.py
 -------------------------------------------------- part of the fastmat package

  Routines for recurring printing and IO tasks


  Author      : wcw
  Introduced  : 2016-04-08
 ------------------------------------------------------------------------------

   Copyright 2016 Sebastian Semper, Christoph Wagner
       https://www.tu-ilmenau.de/ems/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 ------------------------------------------------------------------------------
'''
import sys
import os
import platform
import re

import numpy as np
import scipy as sp

################################################################################
################################################## string styling


class FMT():
    '''Give escape format strings (color, face) a name.'''
    END    = "\033[0m"
    BOLD   = "\033[1m"
    LINE   = "\033[4m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    AQUA   = "\033[96m"


################################################## fmtStr(), fmtCOLOR()
def fmtStr(string, color):
    '''Print a string quoted by some format specifiers.'''
    # colored output only supported with linux
    return "%s%s%s" %(color, string, FMT.END) if platform.system() == 'Linux' \
        else string


def fmtGreen(string):
    '''Print string in green.'''
    return fmtStr(string, FMT.GREEN)


def fmtRed(string):
    '''Print string in red.'''
    return fmtStr(string, FMT.RED)


def fmtYellow(string):
    '''Print string in yellow.'''
    return fmtStr(string, FMT.YELLOW)


def fmtAqua(string):
    '''Print string in aqua.'''
    return fmtStr(string, FMT.AQUA)


def fmtBlue(string):
    '''Print string in blue.'''
    return fmtStr(string, FMT.BLUE)


def fmtBold(string):
    '''Print string in bold face.'''
    return fmtStr(string, FMT.BOLD)


################################################## fmtEscape()
reAnsiEscape = re.compile(r'\x1b[^m]*m')


def fmtEscape(string):
    '''Return a string with all ASCII escape sequences removed.'''
    return reAnsiEscape.sub('', string)


################################################## getConsoleSize()
def getConsoleSize(fallback=(80, 25)):
    size = tuple(int(dim) for dim in os.popen('stty size', 'r').read().split())
    return size if len(size) >= 2 else fallback


################################################################################
################################################## beautiful string framing

def frameLine(
    width=80,
    char='-'
):
    '''Draw one bar of a defined frame.'''
    print(width * char)


def frameText(
    text,
    align='l',
    width=80,
    strBorder=':'
):
    '''Draw one or multiple lines of text within a defined frame.'''
    # print line by line, replace tabs by 8 whitespaces
    for line in text.replace('\t', ' ' * 8).split(os.linesep):
        # determine amount of padding needed
        pad = max(1, width - 2 * len(strBorder) - len(line))

        # padding defaults to left
        l1 = 0
        l2 = max(0, pad - l1)
        if align == 'r':
            l1, l2 = l2, l1
        elif align == 'c':
            l2 = pad // 2
            l1 = pad - l2
        if len(line) > 0:
            print("%s%s%s%s%s" %(
                strBorder, l1 * " ",
                line,
                l2 * " ", strBorder[::-1]
            ))


def printTitle(
    strCaption,
    repChar="-",
    strBorder="||  ",
    width=0,
    style=fmtBold
):
    '''
    helper for printing a very nice box around text :)

        str_caption    - text to be displayed in a box
    '''
    # determine amount of padding needed
    pad = 0 if width < 1 else \
        max(1, width - 2 * len(strBorder) - len(strCaption))

    # pad to width if needed
    strBorder = strBorder + ' ' * (pad // 2)
    if pad % 2 == 1:
        strCaption = strCaption + ' '

    frameLine(char=repChar, width=2 * len(strBorder) + len(strCaption))
    print("%s%s%s" %(strBorder, style(strCaption), strBorder[::-1]))
    frameLine(char=repChar, width=2 * len(strBorder) + len(strCaption))


def printSection(strTitle, newline=True):
    msg = fmtBold(" * %s" %(strTitle))
    if newline:
        print(msg)
    else:
        sys.stdout.write(msg)
        sys.stdout.flush()
