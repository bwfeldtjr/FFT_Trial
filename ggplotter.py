# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:39:35 2019

@author: Brenden
"""
from ggplot import *
import pandas as pd
#The += are required to plot several lines and/or points over each other
def plotter(x,y1,y2=None,y3=None,pt=None,ln=None):
    #If there are three lines and or points to plot
    if y3 is not None:
        data1 = pd.DataFrame({'x':x,'y1':y1,'y2':y2,'y3':y3})
        pltt = ggplot(aes(), data = data1) 
        if pt == 'Yes':
            pltt += geom_point(aes(x='x',y='y1'), data = data1) 
            pltt += geom_point(aes(x='x',y='y2'), data = data1) 
            pltt += geom_point(aes(x='x',y='y=3'), data = data1)
        if ln == 'Yes':
            pltt += geom_line(aes(x='x',y='y1'), data = data1) 
            pltt += geom_line(aes(x='x',y='y2'), data = data1) 
            pltt += geom_line(aes(x='x',y='y3'), data = data1)
    #If there are 2 lines or points
    elif y2 is not None:
        data1 = pd.DataFrame({'x':x,'y1':y1,'y2':y2})
        pltt = ggplot(aes(), data = data1) 
        if pt == 'Yes':
            pltt += geom_point(data1,aes(x='x',y='y1'))
            pltt += geom_point(aes(x='x',y='y2'), data = data1)
        if ln == 'Yes':
            pltt += geom_line(aes('x','y1'), color='blue', data=data1) 
            pltt += geom_line(aes('x','y2'), color='green', data=data1)
    #If there is only one
    else:
        data1 = pd.DataFrame({'x':x,'y1':y1})
        pltt = ggplot(aes(x='x',y='y1'), data = data1) 
        if pt == 'Yes':
            pltt += geom_point()
        if ln == 'Yes':
            pltt += geom_line()

#    print(pltt)
    
    return pltt 