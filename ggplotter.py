# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:39:35 2019

@author: Brenden
"""
from ggplot import *
import pandas as pd

def plotter(x,y1,y2=None,y3=None,pt=None,ln=None):
    if y3 is not None:
        data = pd.DataFrame({'x':x,'y1':y1,'y2':y2,'y3':y3})
        pltt = ggplot(aes(), data = data) 
        if pt == 'Yes':
            pltt += geom_point(aes(x='x',y='y1'), data = data) 
            pltt += geom_point(aes(x='x',y='y2'), data = data) 
            pltt += geom_point(aes(x='x',y='y=3'), data = data)
        if ln == 'Yes':
            pltt += geom_line(aes(x='x',y='y1'), data = data) 
            pltt += geom_line(aes(x='x',y='y2'), data = data) 
            pltt += geom_line(aes(x='x',y='y3'), data = data)
    elif y2 is not None:
        data = pd.DataFrame({'x':x,'y1':y1,'y2':y2})
        pltt = ggplot(aes(), data = data) 
        if pt == 'Yes':
            pltt += geom_point(aes(x='x',y='y1'), data = data) 
            pltt += geom_point(aes(x='x',y='y2'), data = data)
        if ln == 'Yes':
            pltt += geom_line(aes(x='x',y='y1'), data = data, color = 'blue') 
            pltt += geom_line(aes(x='x',y='y2'), data = data, color = 'green')
    else:
        data = pd.DataFrame({'x':x,'y1':y1})
        pltt = ggplot(aes(x='x',y='y1'), data = data) 
        if pt == 'Yes':
            pltt += geom_point()
        if ln == 'Yes':
            pltt += geom_line()
    
#    print(pltt)
    
    return pltt 