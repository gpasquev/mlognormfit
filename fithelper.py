#!/usr/bin/env python
# coding: utf8

import glob

from Tkinter import Tk
from tkFileDialog import askopenfilename



def _safename(name):
    """ Se asegura que el nombre elegido no sobreescriba otro ya existente. """
    n = glob.glob(name+'.*')
    if len(n)>0:
        k=0
        while 1:
            k+=1
            if len(glob.glob(name+'.%2.2d*'%k))==0:
                break
            elif k==99:
                raise Exception, 'Please choose another filename'
        name=name+'.%2.2d'%k
        #print 'The name was changed to "%s" to avoid overwriting another file with the same name.'%name
    else:
        name += '.00'    
    return name


def is_number(s):
    """ Indicates if s is a number (True) or not (False) """
    try:
        float(s)
        return True
    except ValueError:
        return False



def uigetfile():
    """ GUI to select a file. """
    # Robado de http://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return(filename)


