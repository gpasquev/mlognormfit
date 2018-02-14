#!/usr/bin/env python
# coding: utf8


__version__ = 20171003
__author__ = 'GAP'

import numpy as np
import matplotlib.pyplot as pyp

      
def loopf(a,c_ini=-0.5e-8,c_end=-4.5e-8,nump=50,error = 'None',color='r',saveout = False):
    """ 
        Sucesión de ajustes concecutivos, utlizando como parámetro incial el
        resultado del ajuste predecesor. De ajuste a ajsute se va variando la
        constante inicial iniciando en **c_ini**, finalizando en **c_end** y 
        con un número de **nump** pasos.  

        Args
        ----
        a: instancia de :class:`session` del paquete :fit2:.

        Kwargs
        ------
        c_ini: valor de la cosntante inicial inicial
        c_end: valor de la constanta inicail final
        nump:  número de pasos
        error: 'Area' or 'None' (ver fit2)
        color: color del las curvas que visualiza al terminar. 
        saveout: indica si debe guardarse en archivo la salida. 
    """


    # Define peso de los datos.
    a.set_yE_as(error)


    C = np.linspace(c_ini,c_end,nump)
    P = []    # Lista con lmfit.params
    CHI2 = [] # Lista con los Chi2

    print a
    a.plot()
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


    # Parte gráfica 
    pyp.figure(88)
    pyp.plot(C, mum,'.-',color = color,label=a.filename) 
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
    pyp.show()


    if saveout:
        fid = open('../sucesivos_py/'+a.filename+'.'+error+'.sucesivos','w')
        fid.write('# filename: %s\n'%a.filename)
        fid.write('# weigth: %s\n'%error)
        fid.write('#'+80*'-'+'\n')
        fid.write('#%10s %10s %10s %10s %10s\n'%('cte-lin','mu-medio','sigma','N','Chi2'))
        for i in range(len(C)):
            fid.write('%10.6g %10.6g %10.6g %10.6g %10.6g\n'%(C[i],mum[i],sig[i],N[i],CHI2[i])) 
        fid.close()



    


#result = lm.minimize(fitfunc, params, args=(self.X, self.Y,self.EY,self.ndob),ftol=1e-10)

