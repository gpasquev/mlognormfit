#!/usr/bin/env python
# coding: utf8


import sys, re, os
import numpy as np

import lmfit as lm
import matplotlib.pyplot as pyp
import matplotlib.text as matplotlibtext
from matplotlib import gridspec
from scipy.integrate import quad
from scipy.stats import norm

import pygap_work.magne.floglangint as fl

import fithelper as h

# Los ciclos se encuentran en Oe y emu.

__longname__  = 'lognor-langevin with lmfit'
__shortname__ = 'lonolagewi_lmfit'
__version__   = '160223'
__author__    = 'Gustavo Pasquevich'


muB = 9.27400968e-21  # erg/Gauss
kB = 1.3806e-16 #erg/K
pi = np.pi
       

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
    """ La integral númerica a lo bruto: integral entre 0 y infinito con 
        quad.
 
        alpha, mu y sigma tienen que ser escalares.

        Traida de /home/gustavo/Atlantis/FISICA/EXPERIMENTOS/2015_int_num_lognorm_lang"""
    Q = quad(langevin_lognormal, 0, np.inf, 
                                  args=(mu,sigma,alpha), 
                                  #full_output=0, epsabs=1e-4, epsrel=1e-4, 
                                  full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, 
                                  limit=50, points=None, weight=None, wvar=None, 
                                  wopts=None, maxp1=50, limlst=50)[0]
    return Q




def fitfunc(params, x , data = None, eps = None, fastintegral = True, Nmult=1, DS=1e10):
    """ Función para ajuste de múltimpls ciclos utilizando 
        fast-lognormal-langevin-integration (floglangint) 
    
        mult: numero de datasets"""
    T     = params['T__%d'%k].value
    
    ind = range(1,Nmult+1)
    
    xmod = np.mod(x+DS/2.,DS)-DS/2.
    J = np.r_[0,np.where( np.diff(x) > DS/2 )[0],-1]   # j son los indices PREVIOS al salto.
    xmod = np.r_[  x[ J[i] : J[i+1]] for i in xrange( len(J) ) ]

    mu  = np.concatenate([np.empty(j[i+1]-j[i]).fill( params['mu__%d'%k].value ) for i in xrange(len(J))])
    sig = np.concatenate([np.empty(x.shape).fill( params['sig__%d'%k].value ) for k in ind])
    C   = np.concatenate([np.empty(x.shape).fill( params['C__%d'%k].value ) for k in ind])
    N   = np.concatenate([np.empty(x.shape).fill( params['N__%d'%k].value ) for k in ind])

    
    if fastintegral == True:
        alpha = muB*xmod/kB/T
        y = muB*N*fl.integral(alpha,mu,sig) + C*xmod
    else:
        for i in range(len(x)):
            alpha = muB*xmod[i]/(kB*T) 
            y[i] = muB*N*integral(alpha,mu,sig) + C*xmod[i]

    if data is None:
        return y
    else:
        if eps is None:
            return (y - data)
        else:
            return (y - data)/eps
            



class session():
    """ Intento de una session de fiteo. """
    def __init__(self,fname,fitfile=False):
        """ fname:  filename.
            ndob :  number of doublets.
            fitfile: True or {False}.  """
        try: 
            A = np.loadtxt(fname)
        except:
            A = np.loadtxt(fname,skiprows=12)  
        self.filename = os.path.basename(fname)
        self.cfilename = fname
        self.X = A[:,0]
        self.Y = A[:,1]
        try:
            self.EY = A[:,2]
        except:
            self.EY = None

        self.params = lm.Parameters()
        self.params.add('N',   value= 9.88e13,  min=0,vary=1)
        self.params.add('mu',   value= 7.65,  vary=1)
        self.params.add('sig',   value= 1.596,  min=0,vary=1)
        self.params.add('C',   value= -3e-8,  vary=1)
        self.params.add('T', value=300, vary = 0)

    def fit(self):
        self.result = lm.minimize(fitfunc, self.params, args=(self.X, self.Y,self.EY),ftol=1e-10)
        # calculate final result
        if self.EY is None:
            self.Yfit = self.Y + self.result.residual
        else:
            self.Yfit = self.Y + self.result.residual*self.EY
    
        # write error report
        print '='*80
        print 'success:',self.result.success
        print self.result.message
        print lm.fit_report(self.result,show_correl=0)
        self.plot(fitresult = True)

    def print_pars(self,fitresult=False):
        """ Print a list with the parameters """
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params

        print lm.fit_report(params,show_correl=0)

    def getpars(self,fname=None):
        """ Obtiene parámetros de un archivo con resultado de un ajuste."""
        if fname is None:
            fname = h.uigetfile()
        self.oldparams = self.params
        print 'Getting parameters from %s'%fname
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
        print self._snow

    def getpars2(self,a):
        """ Obtiene los parámetros de otra instancia """
        self.params =a.params

    # Manejo de parametros =====================================================
    def fix(self,inn):
        """ If inn is an string fix the parameter inn. """
        self.params[inn].vary = False

    def free(self,inn):
        """ If inn is an string set free the parameter inn. """
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
        self.params[pname].set(value)
        self.plot()

    def update(self):
        """ Actualiza e incorpora el resultado del ajuste como modelo actual """
        self.oldparams = self.params
        self.params = self.result.params


    def plot(self,fitresult=False,numfig=110):
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params
        Yteo = fitfunc(params, self.X)

        pyp.figure(2003)
        pyp.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1],hspace = 0) 

        ax0 = pyp.subplot(gs[0])
        pyp.plot(self.X, self.Y, 'k.-')
        pyp.plot(self.X, Yteo, 'r', lw=2, alpha =0.8)
        ax0.xaxis.set_visible(False)

        ax1 = pyp.subplot(gs[1],sharex=ax0)
        pyp.plot(self.X,self.Y-Yteo,color='gray')
        ax1.yaxis.tick_right()
        ax1.set_axis_bgcolor('#CDCADF')
        ax1.ticklabel_format(style='sci', axis='y')
        ax1.ticklabel_format(useOffset=True)

  
        pyp.legend(loc=0)
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
#        fig = pyp.figure(2000,figsize=(15,7.5))
#        ax = fig.add_axes([AXX, AXY+AXH1, AXW, AXH2])
#        pyp.figure(2000)
#        pyp.cla()
#        pyp.plot(self.X, self.Y, 'k.-')
#        pyp.plot(self.X, Yteo, 'r', lw=2, alpha =0.8)
#        ax2 = fig.add_axes([AXX, AXY, AXW, AXH1])
#        pyp.plot(self.X,self.Y-Yteo,color='gray')


        pyp.figure(2001)
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
        fig.text(AXX,AXY+AXH+AXY/4.,os.path.basename(self.fname))
        if outfname is not None:
            fig.text(AXX,AXY+AXH+2*AXY/4.,os.path.basename(outfname))
        fig.text(1-AXX/2,0,'%s ver%s'%(__shortname__,__version__),horizontalalignment='right',color='k',style='italic')

    def save(self,outfname=None,outfig=True):
        """ Save to file fitting result (if it exist) """

        # la idea sería grabar en una carpeta que se encuentra un nivel más abajo de donde
        # obtuvo el archivo para ajustar. O mejor (más fácil) preguntar en que carpeta guardar.
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
            raise ValueError,'Aún no implementado'            
        print fitpardir
        print 'outputfilename: %s'%outfname

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

def new():
    fname = h.uigetfile()    
    a = session(fname)
    a.plot()
    return a 


#a = session('ciclos/121518centri100(25)4.txt')

#result = lm.minimize(fitfunc, params, args=(self.X, self.Y,self.EY,self.ndob),ftol=1e-10)

