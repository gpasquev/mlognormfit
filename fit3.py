#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Este es un módulo que tiene herramientas para ajustar un ciclo M vs H con 
    una distribución lognormal de Langevins. 

    ======================
    How to use it
    ======================

    The module provide a function :func:`new` to load data from a file and 
    create an instance of class:`session`. Otherwise you can init a seasson 
    directly with the class initialization method.

    ===========
    Modo de uso
    ===========
    Tiene una clase donde se gestiona el ajuste class:`session`.

    ============
    Dependencias
    ============

    Dependencias comunes
    --------------------
    * Tiene un módulo fithelper.py que debe acompañarlo.
    * Utiliza matplotlib para hacer gráficos.
    * Utiliza numpy.
    * Utiliza scipy. 
    
    Dependencia menos comunes
    -------------------------
    * Utiliza el paquete lmfit como motor de ajuste. 
    * Utiliza floflangint (módulo de cálculo rápido de la 
        integral de Langevin*lognormal)



"""

import os, re
import numpy as np

import lmfit as lm
import matplotlib.pyplot as pyp
from matplotlib import gridspec
from scipy.integrate import quad
import uncertainties as un
import uncertainties.unumpy as unumpy

# local lmodules
from . import  floglangint as fl
from . import  fithelper as h




# Los ciclos se encuentran en Oe y emu.

__longname__  = 'several lognor-langevin with lmfit'
__shortname__ = 'multilonolagewi_lmfit'
__version__   = '190506'
__author__    = 'Gustavo Pasquevich'


muB = 9.27400968e-21  # erg/Gauss
kB  = 1.3806e-16 #erg/K
pi  = np.pi
       

def langevin(x,a=1.):
    """ Langevin function.
        .. math::
            langevin(x;a) = L(a*x) = coth(a*x) - 1/(a*x)
    """
    w = x==0
    if np.any(w):                   #  Los condicionales aparecen para evitar la 
        w2 = ~w                              #  singularidad removible. Ver nota
        y = np.zeros(np.shape(x))                     #  150322 en el historial.
        y[w2] = 1./np.tanh(a*x[w2]) -1./(a*x[w2])
        return y
    else:      
        return 1./np.tanh(a*x) -1./(a*x)
def langevin_lognormal(x,mu,sigma,alpha):
    """ Lognormal distribution of Langevins functions. *x0* and *s* are the 
        parameters defining the Log-Normal distribution.

        f(x)*x*Lan(a*x)

        31/10/15 cambio de llamado lognormal a forma explicita para poder eliminar
                 la variable x que aparece multiplicada y dividida."""
    return 1./np.sqrt(2.*pi)/sigma*np.exp(-(np.log(x)-mu)**2./(2.*sigma**2.))*langevin(x,alpha)

def integral(alpha,mu,sigma):
    """ La integral númerica **a lo bruto**: integral entre 0 y infinito con 
        quad.
 
        alpha, mu and sigma should be numbers (not arrays).

        Taken from /home/gustavo/Atlantis/FISICA/EXPERIMENTOS/2015_int_num_lognorm_lang"""
    Q = quad(langevin_lognormal, 0, np.inf, 
                                  args=(mu,sigma,alpha), 
                                  #full_output=0, epsabs=1e-4, epsrel=1e-4, 
                                  full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, 
                                  limit=50, points=None, weight=None, wvar=None, 
                                  wopts=None, maxp1=50, limlst=50)[0]
    return Q


def fitfunc(params,x,data = None, eps = None,nint=1,fastintegral = True):
    """ Fitting function prepared for lmfit. It fits a lognormal distribution
        of Langevins. """

    y = np.zeros(len(x))
    C     = params['C'].value
    dc    = params['dc'].value
    T     = params['T'].value
    
    for k in range(nint):
        N     = params['N%d'%k].value
        mu    = params['mu%d'%k].value
        sig   = params['sig%d'%k].value
        alpha = muB*x/kB/T
        y += muB*N*fl.integral(alpha,mu,sig)
    
    y += dc + C*x
    
    if data is None:
        return y
    else:
        if eps is None:
            return (y - data)
        else:
            return (y - data)/eps
            

def maglognormlangevin(x,N,mu,sig,T):
    """ Integral of langevin-functions lognormal-distribuition.
    
        Function prepared to be used as a Model function of lmft Model class.
        This function uses fast lognormal-langevin calculation (floglangint).
    """
    alpha = muB*x/(kB*T) 
    y = muB*N*fl.integral(alpha,mu,sig)
    return y 

def lineal(x,C,dc):
    return dc + C*x

class session():
    """ Attempt to a fitting session.
    
        X, Y , EY: unidimensional arrays with magentic field (X), magnetization 
	               (or magnetic moment) (Y), and error (or fitting wheigt) (EY).
    

    """
    def __init__(self,X,Y,EY = None,fitfile=False,mass=None,fname='proof'):
        """ X, Y , EY: unidimensional arrays with magentic field (X), 
                   magnetization (or magnetic moment) (Y), 
                   and error (or fitting wheigt) (EY).
	        mass:  mass. Is used to convert input data from emu to emu/g. 
	               If mass is given, is used to divide **Y** by mass. 
                   If mass is None (not givene) nothing happens.
  
            fname:  filename. // I don't know what must do this variable now!!!
            fitfile: True or {False} .  PARECE UNA VARIABLE INUTIL
            mass: {None} or number. If None  
        """

        self.filename = os.path.basename(fname)
        self.cfilename = fname
        self.mass = mass
        self.X = X
        self.Y = Y
        self.EY = EY
        self.nint  = 1
        if EY == None:
            self.EYkind = 'None'
        else:
            self.EYkind = '3rd-col'

        if mass is not None:
            self.Y = self.Y/mass
            if self.EYkind == '3rd-col':
                Warning('The "mass" is not affecting the error column')


        self.params = lm.Parameters()
        self.params.add('N0',   value= 9.88e13,  min=0,vary=1)
        if mass is not None:
            self.params['N0'].set(9.88/self.mass)
        self.params.add('mu0',   value= 7.65,  vary=1)
        self.params.add('sig0',   value= 1.596,  min=0,vary=1)
        self.params.add('C',   value= -3e-8,  vary=1)
        self.params.add('T', value=300, vary = 0)
        self.params.add('dc', value=0, vary = 0)

    def addint(self):
        """ Add a new lognormal distribution to fitting model """
        n = self.nint
        self.params.add('N%d'%n,   value= self.params['N%d'%(n-1)].value/10,  min=0,vary=1)
        self.params.add('mu%d'%n,   value= 7.65,  vary=1)
        self.params.add('sig%d'%n,   value= 1.596,  min=0,vary=1)
        self.params.add('C',   value= -3e-8,  vary=1)
        self.params.add('T', value=300, vary = 0)
        self.nint += 1
        
    def set_yE_as(self,kind):
        """ Set the type of weight used in the MvsH-curve fitting.
            
            Parameters
            ----------
            kind:
                'sep': 
                    set weigth inverse to x-difference between points.
                    'Area', si accepetd as same value for this argument. 
                    'Area' was inherited from previus versions. 
                'None' or None: 
                    set uniform weight. 
                
                 
        """
        if kind == 'sep' or kind == 'area' or kind == 'Area':
            D = np.diff(self.X)
            D1 = np.append(D[0],D)
            D2 = np.append(D,D[-1])
            D = (D1+D2)/2.+1.23456e-10 
            self.EY = 1/D
            self.EYkind = 'sep'
        if kind == 'None' or kind == None:
            self.EY = None
            self.EYkind = 'None'

    


    def fit(self):
        self.result = lm.minimize(fitfunc, self.params, 
                                  args=(self.X, self.Y,self.EY,self.nint),
                                  ftol=1e-10)
        # calculate final result
        if self.EY is None:
            self.Yfit = self.Y + self.result.residual
        else:
            self.Yfit = self.Y + self.result.residual*self.EY
    
        # write error report
        print ('='*80)
        print ('success:',self.result.success)
        print (self.result.message)
        print (lm.fit_report(self.result,show_correl=0))
        self.plot(fitresult = True)

    def print_pars(self,fitresult=False):
        """ Print a list with the parameters """
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params



        print (lm.fit_report(params,show_correl=0))

        Ns    = [params['N%d'%i].value for i in range(self.nint)]
        eNs   = [params['N%d'%i].stderr for i in range(self.nint)]
        mus   = [params['mu%d'%i].value for i in range(self.nint)]
        emus  = [params['mu%d'%i].stderr for i in range(self.nint)]
        sigs  = [params['sig%d'%i].value for i in range(self.nint)]
        esigs = [params['sig%d'%i].stderr for i in range(self.nint)]

        def mapf(x):
            if x == None:
                return 0
            else:
                return x
        emus  = list(map(lambda x: mapf(x),emus))
        esigs = list(map(lambda x: mapf(x),esigs))
        eNs   = list(map(lambda x: mapf(x),eNs))

        mus  =  np.array([un.ufloat(mus[i] ,emus[i])   for i in range(self.nint)])
        sigs =  np.array([un.ufloat(sigs[i],esigs[i]) for i in range(self.nint)])
        Ns   =  np.array([un.ufloat(Ns[i]  ,eNs[i])     for i in range(self.nint)])

        mums  = unumpy.exp(mus)*unumpy.exp(sigs**2/2.)
        #        emums = mums*emus + sigs*mums*esigs 
        SD = mums*unumpy.sqrt( unumpy.exp(sigs**2) - 1 )
        
        mmuN = mums*Ns/sum(Ns)  # mean-mu-N
        
        print('-------------------------------')
        print('mean-mu      = %s mb'%(mmuN[0]))
        print('standard-dev = %s mb'%SD[0]) 
        print('- - - - - - - - - - - - - - - -')
        #print('lognorm-sig  = %'%sig) 


    def getpars(self,fname=None):
        """ Get parameters from file."""
        if fname is None:
            fname = h.uigetfile()
        self.oldparams = self.params
        print ('Getting parameters from %s'%fname)
        fid = open(fname)
        A  = fid.read()
        a  = re.search('\[\[Variables\]\][\n\s\W\w]*?\[\[',A)
        a1 = a.group()    
        gg = re.findall('\s*(\S*):\s*(\S*)',a1)
        newparams = lm.Parameters()       
        for k in gg:
            if h.is_number(k[1]):
                newparams.add(k[0],float(k[1]))
        self.params = newparams

        self.Ynow = fitfunc(self.params,self.X)
        self.plot()
        self._snow = sum(self.Ynow)
        print (self._snow)

    def getpars2(self,a):
        """ Obtiene los parámetros de otra instancia """
        self.params =a.params

    # Manejo de parametros =====================================================
    def fix(self,inn):
        """ Fix the parameter inn (inn must be the parameter-name). """
        self.params[inn].vary = False

    def free(self,inn):
        """ Set the parameter named "inn" as free. """
        self.params[inn].vary = True

    def plink(self,p1,expr):
        """ Setea el parametro p1 como expresion.
            p1 un string con el nombre del parámetro.
            expr: un string con la expresion. """
        self.params[p1].expr=expr

    def setfree(self,pname):
        self.params[pname].vary = True
    
    def setfix(self,pname):
        self.params[pname].vary = False

    def setp(self,pname,value):
        """ Set the **value** to the parameter **pname**. """
        self.params[pname].set(value)
        self.plot()

    def update(self):
        """ Actualiza e incorpora el resultado del ajuste como modelo actual """
        self.oldparams = self.params
        self.params = self.result.params


    def plot(self,fitresult=False,numfig=110,typediff='abs'):
        """ plot curve, model, nand difference. """
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params
        Yteo = fitfunc(params, self.X,nint=self.nint)

        pyp.figure(2003)
        pyp.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1],hspace = 0) 

        ax0 = pyp.subplot(gs[0])
        pyp.plot(self.X, self.Y, 'k.-')
        pyp.plot(self.X, Yteo, 'r', lw=2, alpha =0.8)
        ax0.xaxis.set_visible(False)

        ax1 = pyp.subplot(gs[1],sharex=ax0)
        if typediff == 'abs':
            pyp.plot(self.X,self.Y-Yteo,color='gray') 
        elif typediff == 'rel':
            pyp.plot(self.X,self.Y/Yteo-1,color='gray')

        ax1.yaxis.tick_right()
        ax1.set_facecolor('#CDCADF')
        ax1.ticklabel_format(style='sci', axis='y')
        ax1.ticklabel_format(useOffset=True)
 
        pyp.xlabel('X')
        pyp.ylabel('Y')
        #pyp.xlim([-5,5])

    def plot2(self,outfname = None,fitresult=False):
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params
        Yteo = fitfunc(params, self.X)

        FIGSIZEX, FIGSIZEY  = 15,7.5
        AXX, AXY, AXH1, AXH2, AXW = 0.1, 0.1, 0.1, 0.7, 0.4
        AXH = AXH1 + AXH2


        fig = pyp.figure(2001)
        ax1 = pyp.subplot(2,1,1)
        pyp.cla()
        pyp.plot(self.X, self.Y, 'k.-')
        pyp.plot(self.X, Yteo, 'r', lw=2, alpha =0.8)

        ax2 = pyp.subplot(2,1,2,sharex=ax1)
        #ax2 = fig.add_axes([AXX, AXY, AXW, AXH1])
        pyp.plot(self.X,self.Y-Yteo,color='gray')

        pyp.subplots_adjust(left=AXX, bottom=AXY, right=AXX+AXW, top=AXY+AXH1+AXH2,
                wspace=None, hspace=0)

        if fitresult:
            ptext = lm.fit_report(self.result,show_correl=0)
        else:
            ptext = lm.fit_report(self.params,show_correl=0)
            
        fig.text(AXX+AXW+AXX/2.,AXY+AXH,ptext,verticalalignment='top')        
        #        fig.text(AXX,AXY+AXH+AXY/4.,os.path.basename(self.fname))
        if outfname is not None:
            fig.text(AXX,AXY+AXH+2*AXY/4.,os.path.basename(outfname))
        fig.text(1-AXX/2,0,'%s ver%s'%(__shortname__,__version__),horizontalalignment='right',
                 color='k',style='italic')

    def save(self,outfname=None,outfig=True):
        """ Save to file fitting result (if it exist) """

        # la idea sería grabar en una carpeta que se encuentra un nivel mas 
        # abajo de donde obtuvo el archivo para ajustar. O mejor (más fácil) 
        # preguntar en que carpeta guardar.
        if outfname is None:
            # Se fija si existe una carpeta fits en el path del archivo de entrada.
            # Si no es así, la crea. Luego guarda con safename en esa carpeta el resultado del
            # ajuste con la extensión .fit y un número protector.  
            dirr = os.path.dirname(self.fname)
            fname = os.path.basename(self.fname)
            fitpardir = os.path.join(dirr,'fits/')
            if not os.path.exists(fitpardir):
                os.makedirs(fitpardir)
            outfname = h._safename(os.path.join(fitpardir,fname + '.fit'))
        else:
            raise (ValueError,'Aún no implementado')            
        print(fitpardir)
        print('outputfilename: %s'%outfname)

        fid = open(outfname,'w')
        fid.write('script internal name (ver:%s): %s\n'%(__version__,__shortname__))    
        fid.write('data-filename:%s\n'%self.fname)
        fid.write('this-filename:%s\n'%outfname)
        fid.write('[[status]]\n')
        fid.write('success:%s\n'%self.result.success)
        fid.write(self.result.message+'\n')
        fid.write(lm.fit_report(self.result)+'\n')

        # Print Data, model and contributions curves----------------------------
        fid.write('[[data]]\n')
        A = []
        A.append(self.X)
        A.append(self.Y)
        if self.EY is not None:
            A.append(self.EY)
        YFIT = fitfunc(self.result.params, self.X)
        A.append(YFIT)

        A = np.array(A).T
        nc,nf = np.shape(A)
        for i in range(nc):
            for j in range(nf):
                fid.write('%e '%A[i,j])
            fid.write('\n')
        fid.close()

        if outfig == True:
            pyp.ioff()
            self.plot2(outfname = outfname,fitresult=True)
            pyp.savefig(outfname+'.png',dpi =300)
            pyp.close()
            pyp.ion()
            
        

def new(fname=None,mass = None,label=None,**kwarg):
    """ **kwargs are passed directly to numpy.loadtxt """
    if fname == None:
        fname = h.uigetfile()
    A = np.loadtxt(fname,**kwarg)
    x = A[:,0]
    y = A[:,1]    
    a = session(x,y,mass = mass, fname =fname)
    a.plot()
    return a 




