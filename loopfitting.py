#!/usr/bin/env python
# coding: utf8

""" Script python para realizar ajustes sucesivos sobre un ciclo variando 
    la constante dia/paramagnétcia de la contribución lineal.

    VARIABLES DE INICIALIZACIÓN
    ===========================
    En el script se deben indicar las siguientes variables.

    fname: mombre del archivo con la medida del VSM.
    error: string 'area' o 'None'. 
                'None' indica que el "error" que va a utilizarse para la 
                magentización es ninguno, es decir todo los datos tienen la misma
                importancia.
                'area' indica que el "error" va a ser tal que el peso de los datos
                este dado por el área que encierran. [Página 118 cuaderno Gustavo]
    mass: None o un número. De existir el número didirá la columna por la masa 
          antes de empezar.
    c_ini: constante lineal inicial.
    c_end: constante lineal final.
    nump: número de pasos.


"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as pyp

import fit2

color = 'g'
label = None
fname = '161107sobre_17.txt'
mass = 54.0e-6 #L
error = 'area'
c_ini = 8e-5
c_end = -1e-4
nump  = 200


a = fit2.session('../ciclos/'+fname,mass = mass)  
a.setp('N',1e18)
a.set_yE_as(error)

C = np.linspace(c_ini,c_end,nump)
P = []    # Lista con lmfit.params
CHI2 = [] # Lista con los Chi2


a.fit()
a.update()
for c in C:
    a.setp('C',c)
    a.fix('C')
    a.fit()
    a.update()
    P.append(a.params)
    CHI2.append(a.result.chisqr)
CHI2 = np.array(CHI2)
mu0 = np.array([p['mu'].value for p in P])
sig = np.array([p['sig'].value for p in P])
N   = np.array([p['N'].value for p in P])
mum = np.exp(mu0+sig**2/2.)


pyp.figure(88)
pyp.plot(C, mum,'.-',color = color,label=fname) 
pyp.title('mumedio')

pyp.figure(89)
pyp.plot(C,sig,'.-',color=color)
pyp.title('sigma') 
pyp.figure(90)
pyp.plot(C,N,'.-',color=color)
pyp.title('N')

pyp.figure(91)
pyp.plot(C,CHI2,'.-',color=color)
pyp.title('Chi2')

fid = open('../sucesivos_py/'+fname+'.'+error+'.sucesivos','w')
fid.write('# filename: %s\n'%fname)
fid.write('# weigth: %s\n'%error)
fid.write('#'+80*'-'+'\n')
fid.write('#%10s %10s %10s %10s %10s\n'%('cte-lin','mu-medio','sigma','N','Chi2'))
for i in range(len(C)):
    fid.write('%10.6g %10.6g %10.6g %10.6g %10.6g\n'%(C[i],mum[i],sig[i],N[i],CHI2[i])) 
fid.close()


pyp.show()




#result = lm.minimize(fitfunc, params, args=(self.X, self.Y,self.EY,self.ndob),ftol=1e-10)

