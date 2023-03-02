#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    This module provides tools to fit M vs H cycles with langevin 
    lognormal-distribution.  

    ======================
    How to use it
    ======================

    Assuming that the matrices **x** and **y** correspond to a 
    magnetic anhysteretic curve (**x** field and **y** magnetic moment or 
    magnetization), the fitting session can be initialized as follows:  
    
        s = session(**x**, **y**) 
    
    This module also provides a function :func:`new` to load data from a file 
    and create an instance of class:`session`. The file is opened with 
    numpy.loadtxt. It acn be passed kwargs to loadtxt. See :func:`new` 
    docstring: 
        
        s = new(fname= 'fname.txt',**loadtxt_kwargs)
    
    or using a browser windows to coose the input file: 
    
        s = new(**loadtxt_kwargs)
    

    See :class:`session` documentation to see how to work with a session fit.   


    ============
    Dependencies
    ============

    usual Dependencies 
    --------------------
    * matplotlib (graphs).
    * numpy.
    * scipy. 
    
    Not so usual Dependencies 
    -------------------------
    * lmfit 
    * uncertainties



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

__longname__  = 'multi lognormal-langevin fit with lmfit'
__shortname__ = 'multilognorlanglmfit'
__version__   = '210707'
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
    if np.any(w):                  #  Los condicionales aparecen para evitar la 
        w2 = ~w                             #  singularidad removible. Ver nota
        y = np.zeros(np.shape(x))                    #  150322 en el historial.
        y[w2] = 1./np.tanh(a*x[w2]) -1./(a*x[w2])
        return y
    else:      
        return 1./np.tanh(a*x) -1./(a*x)
    
def lognorm(x,mu,sigma):    
    return 1./np.sqrt(2.*pi)/x/sigma*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
    
def langevin_lognormal(x,mu,sigma,alpha):
    """ Lognormal distribution of Langevins functions. *x0* and *s* are the 
        parameters defining the Log-Normal distribution.

        f(x)*x*Lan(a*x)
    """
    return 1./np.sqrt(2.*pi)/sigma*np.exp(-(np.log(x)-mu)**2./(2.*sigma**2.))*langevin(x,alpha)

def integralq(alpha,mu,sigma):
    """ numeric integral  **a lo bruto**: integral entre 0 y infinito con 
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


def fitfunc(params,x,data = None, eps = None,ndist=1,fastintegral = True):
    """ Fitting function prepared for lmfit. It fits a lognormal distribution
        of Langevins. 
        
        kwarg
        =====
        fastintegral: -True (default), it use fast-lognormal-langvin integration.
                      -False (not yet implemented), numeric integral should be 
                      done with quad (todo!!!)
    """

    y = np.zeros(len(x))
    C     = params['C'].value
    dc    = params['dc'].value
    T     = params['T'].value
    
    for k in range(ndist):
        N     = params['N%d'%k].value
        mu    = params['mu%d'%k].value
        sig   = params['sig%d'%k].value
        alpha = muB*x/kB/T
        if fastintegral:
            y += muB*N*fl.integral(alpha,mu,sig)
        else:
            y += muB*N*np.array([integralq(a,mu,sig) for a in alpha])
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

class session():
    """ A fitting session.
    
        X, Y , EY: unidimensional arrays with magentic field (X), magnetization 
	               (or magnetic moment) (Y), and error 
                   (or fitting wheigt) (EY).
    
        s.fit()     fit
        s.addint()  add a log-normal distribution.
        
    """
    fastintegral = True
    def __init__(self,X,Y,EY = None,mass=None,fname='proof',divbymass = False):
        """ X, Y , EY: unidimensional arrays with magentic field (X), 
                   magnetization (or magnetic moment) (Y), and error (or 
                   fitting weight) (EY).
	        
            mass:  {None}, a number or a uncertainties.ufloat (see divbymas). 
               
                    It is used to convert input data from emu to emu/g 
                    (if dovbymass = True) or to calculate Ms (if 
                    divbymass = False)

                    If mass is None (not given) nothing happens.
                    
                    if mass is a two values tuple: (m1,m2), it correspond to
                    to a mass with uncertainty: m1 +/- m2 
    
            fname:  filename.
         
                   
            divbymass: boolean value inidcating if **Y** should be divided by 
                   **mass** before fitting or not. 
                   
                   This division is evaluated only if **mass** is not None. 
                   If **divbymass** is False, the fitting is done over 
                   **Y** as entered. In that case the **mass** is used only 
                   for Ms calculation.
                   When used only for Ms calculation, **mass** can be given as 
                   an uncertainties.ufloat instance. 
                   
        """

        self.filename = os.path.basename(fname)
        self.cfilename = fname
        
        if type(mass) == type(tuple()):
            mass = un.ufloat(mass[0],mass[1])
        self.mass = mass
        self.X = X
        self.Y = Y
        self.EY = EY
        self.ndist  = 1
        self.divbymass = divbymass
        if EY == None:
            self.EYkind = 'None'
        else:
            self.EYkind = '3rd-col'

        if mass is not None and divbymass:
            self.Y = self.Y/mass
            if self.EYkind == '3rd-col':
                Warning('The "mass" is not affecting the error column')

        self.params = lm.Parameters()
        self.params.add('N0',   value= 9.88e13,  min=0,vary=1)
        if mass is not None and divbymass:
            self.params['N0'].set(9.88e13/self.mass)
        self.params.add('mu0',   value= 7.65,  vary=1)
        self.params.add('sig0',   value= 1.596,  min=0,vary=1)
        self.params.add('C',   value= -3e-8,  vary=1)
        self.params.add('T', value=300, vary = 0)
        self.params.add('dc', value=0, vary = 0)

    def addint(self,guess=True):
        """ Add a new lognormal distribution to fitting model."""
        # initial values: mu  mu(n-1). sig = sig(n-1). N = N(n-1)/10.
        
        n = self.ndist
        self.params.add('N%d'%n, value= self.params['N%d'%(n-1)].value/10,
                                                      min=0,vary=1)
        self.params.add('mu%d'%n, value= self.params['mu%d'%(n-1)].value,  
                                                            vary=1)
        self.params.add('sig%d'%n, value= self.params['sig%d'%(n-1)].value, 
                                                      min=0,vary=1)

        self.ndist += 1
        
    def set_yE_as(self,kind):
        """ Set the type of weight used in the MvsH-curve fitting.
            
            Parameters
            ----------
            kind:
                'sep': 
                    set weight inverse to x-difference between points.
                    'Area', is accepetd as same value for this argument. 
                    ('Area' was inherited from previous versions.) 
                'None' or None: 
                    set uniform weight. 
                
                 
        """
        if kind == 'sep' or kind == 'area' or kind == 'Area':
            D = np.diff(self.X)
            D1 = np.append(D[0],D)
            D2 = np.append(D,D[-1])
            D = (D1+D2)/2.+1.23456e-10    # <======== 
            self.EY = 1/D
            self.EYkind = 'sep'
        elif kind == 'None' or kind == None:
            self.EY = None
            self.EYkind = 'None'
        else:
            raise ValueError('%s is not a valid "kind" string. \
                             try "sep" or "None"'%kind)
    


    def fit(self):
        self.result = lm.minimize(fitfunc, self.params, 
                                  args = (self.X, self.Y,self.EY,self.ndist),
                                  ftol = 1e-10 )
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

    def print_pars(self,fitresult=False,ret= False):
        """ Print a list with the parameters and results 
        
            ret: if True, it does not print in screen and returns ouput string 
        """
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params

        
        Ns    = [params['N%d'%i].value for i in range(self.ndist)]
        eNs   = [params['N%d'%i].stderr for i in range(self.ndist)]
        mus   = [params['mu%d'%i].value for i in range(self.ndist)]
        emus  = [params['mu%d'%i].stderr for i in range(self.ndist)]
        sigs  = [params['sig%d'%i].value for i in range(self.ndist)]
        esigs = [params['sig%d'%i].stderr for i in range(self.ndist)]

        def mapf(x):
            """ Internal function to set cero error to fixed parameters. """
            if x == None:
                return 0
            else:
                return x
        
        emus  = list(map(lambda x: mapf(x),emus))
        esigs = list(map(lambda x: mapf(x),esigs))
        eNs   = list(map(lambda x: mapf(x),eNs))

        mus  = np.array([un.ufloat(mus[i] ,emus[i]) for i in range(self.ndist)])
        sigs = np.array([un.ufloat(sigs[i],esigs[i]) for i in range(self.ndist)])
        Ns   = np.array([un.ufloat(Ns[i]  ,eNs[i])  for i in range(self.ndist)])

        mums  = unumpy.exp(mus)*unumpy.exp(sigs**2/2.)  #list of <mu>_number
        #        emums = mums*emus + sigs*mums*esigs 
        mu2ms = unumpy.exp(2*mus)*unumpy.exp(2*sigs**2) #list of <mu^2>_number

        mmuN  = sum(mums*Ns)/sum(Ns)                    # total <mu>_number
        mmu2N = sum(mu2ms*Ns)/sum(Ns)                   # total <mu^2>_number 
        SD = unumpy.sqrt(mmu2N - mmuN**2)
        mmumu = mmuN*((SD/mmuN)**2+1)                   # total <mu>_mu
        N     = sum(Ns)                                 # total N
        
        Ms = muB*sum(Ns*mums)
        
        Yteo = fitfunc(params, self.X,ndist=self.ndist)
        ssqua = sum((self.Y-Yteo)**2)
        

        # in ot (output-text) it will building the output text
        ot = lm.fit_report(params,show_correl=0)
        ot += '\n' 
        ot +='[[Derived Parameters]]\n'
        ot +='    mean-mu      = {:.4uf} mb\n'.format(mmuN)
        ot +='    stddev       = {:.4uf} mb\n'.format(SD)
        ot +='    <mu>_mu      = {:.4uf} mb\n'.format(mmumu)
        ot +='    sum squares  = %.6e\n'%ssqua
        ot +='    m_s          = {:.4ue} (units)\n'.format(Ms)
        if self.mass is not None and not self.divbymass:
            ot +='    m_s/mass     = {:.4ue} (units)\n'.format(Ms/self.mass)
            if type(self.mass) == type(mmuN): # i.e. of is ufloat
                ot +=19*' '+'(where mass = {:.4ue})\n'.format(self.mass)
            else:
                ot +=19*' '+'(where mass = {:.4e})\n'.format(self.mass)
                
                
        #ot +='- - - - - - - - - - - - - - - -\n'
        #print('lognorm-sig  = %'%sig) 

        if ret:
            return ot
        else:
            print(ot)


    def getpars(self,fname=None):
        """ Get parameters from file."""
        if fname is None:
            fname = h.uigetfile()
        self.oldparams = self.params
        print ('Getting parameters from %s'%fname)
        fid = open(fname)
        A  = fid.read()
        
        # I have use https://regex101.com/ to test and 
        # desingn regular next expressions. 
        a  = re.search('\[\[Variables\]\][\n\s\W\w]*?\[\[',A)
        a1 = a.group()    
        gg = re.findall('\s*(\S*):\s*(\S*)\s(\S*)',a1)
        newparams = lm.Parameters()       

        for k in gg:
            if h.is_number(k[1]):
                if k[2] == '(fixed)':
                    vary = 0
                else:
                    vary = 1
                newparams.add(k[0],float(k[1]),vary=vary)
        self.params = newparams

        self.Ynow = fitfunc(self.params,self.X)
        self.plot()
        self._snow = sum(self.Ynow)
        print (self._snow)

    def getpars2(self,a):
        """ Takes parameters from other instance """
        self.params = a.params.copy()

    # Manejo de parametros ====================================================
    def fix(self,inn):
        """ Fix the parameter inn (inn must be the parameter-name). """
        self.params[inn].vary = False

    def free(self,inn):
        """ Set the parameter named "inn" as free. """
        self.params[inn].vary = True

    def plink(self,p1,expr):
        """ Defines expression for given parameter.
        
            p1: string with parameter name.
            expr: expresion string. """
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
        """ It updates fitting results as actual model. """
        self.oldparams = self.params
        self.params = self.result.params


    def plot(self,fitresult=False,numfig=110,typediff='abs'):
        """ plot curve, model, nand difference. """
        if fitresult == True:
            params = self.result.params
        else:
            params = self.params
        Yteo = fitfunc(params, self.X,ndist=self.ndist)

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

        pyp.subplots_adjust(left=AXX, bottom=AXY, right=AXX+AXW, 
                            top=AXY+AXH1+AXH2,
                            wspace=None, hspace=0)

        if fitresult:
            ptext = lm.fit_report(self.result,show_correl=0)
        else:
            ptext = lm.fit_report(self.params,show_correl=0)
            
        fig.text(AXX+AXW+AXX/2.,AXY+AXH,ptext,verticalalignment='top')        
        #        fig.text(AXX,AXY+AXH+AXY/4.,os.path.basename(self.fname))
        if outfname is not None:
            fig.text(AXX,AXY+AXH+2*AXY/4.,os.path.basename(outfname))
        fig.text(1-AXX/2,0,'%s ver%s'%(__shortname__,__version__), 
                           horizontalalignment='right',
                           color='k',style='italic')

    def save(self,outfname=None,outfig=True):
        """ Save to file fitting result (if it exist).
        
            Saves results of last fit. 
            It runs the same way whether method 'update' is executed or not. 
        """

        # la idea sería grabar en una carpeta que se encuentra un nivel mas 
        # abajo de donde obtuvo el archivo para ajustar. O mejor (más fácil) 
        # preguntar en que carpeta guardar.
        if outfname is None:
            # Se fija si existe una carpeta fits en el path del archivo de 
            # entrada. Si no es así, la crea. Luego guarda con safename en esa 
            # carpeta el resultado del ajuste con la extensión .fit y un 
            # número protector.  
            dirr = os.path.dirname(self.filename)
            fname = os.path.basename(self.filename)
            fitpardir = os.path.join(dirr,'fits/')
            if not os.path.exists(fitpardir):
                os.makedirs(fitpardir)
            outfname = h._safename(os.path.join(fitpardir,fname + '.fit'))
        else:
            raise (ValueError,'Aún no implementado')            
        print(fitpardir)
        print('outputfilename: %s'%outfname)

        fid = open(outfname,'w')
        fid.write('script internal name (ver:%s): %s\n'%(__version__,
                                                         __shortname__))    
        fid.write('data-filename:%s\n'%self.filename)
        fid.write('this-filename:%s\n'%outfname)
        fid.write('[[Definition parameters]]\n')
        fid.write('ndist: %d\n'%self.ndist)
        fid.write('weight: %s\n'%self.EYkind)
        fid.write('[[status]]\n')
        fid.write(self.result.message+'\n')
        fid.write(lm.fit_report(self.result)+'\n')
        
        # Print the Derived Parameters
        printtxt = self.print_pars(fitresult=True,ret=True)
        fid.write('[[Derived Parameters]]')
        fid.write(printtxt.split('[[Derived Parameters]]')[1])

        # Print Data, model and contributions curves---------------------------
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
    """ get x-y data form file.
    
        **kwargs are passed directly to numpy.loadtxt. 
    """
    if fname == None:
        fname = h.uigetfile()
    A = np.loadtxt(fname,**kwarg)
    x = A[:,0]
    y = A[:,1]    
    a = session(x,y,mass = mass, fname =fname)
    a.plot()
    return a 


def plotdist(c,xmax=1e5,cla=True,label=None,axes=None):
    """ Temporal function , to be moved inside session class.""" 
    mu = list()
    sig = list()
    mum = list()
    N = list()
    
    x = np.linspace(1,xmax,100000)
    y = np.zeros((len(x),c.ndist))
    
    for k in range(c.ndist):
        mu.append(c.params['mu%d'%k].value)
        sig.append(c.params['sig%d'%k].value)
        N.append(c.params['N%d'%k].value)
        mum.append(np.exp(mu[-1])*np.exp(sig[-1]**2/2))
        y[:,k] = N[k]*lognorm(x,mu[k],sig[k])/sum(N)
        
    mu = np.array(mu)
    sig = np.array(sig)
    N = np.array(N)


    if axes == None:
        fig1 = pyp.figure(993)
        ax1 = pyp.axes()
        fig1 = pyp.figure(994)
        ax2 = pyp.axes()
    else:
        ax1,ax2 = axes
        
    if cla: 
        ax1.cla()
        ax2.cla()
    ax1.set_title('number distribution')
    ax2.set_title('$\mu$ distribution')
 
    if c.ndist > 1:
        for k in range(c.ndist):
            ax1.plot(x,y[:,k],'--',label=label)
    ax1.plot(x,np.sum(y,1),label=label)
    ax1.legend(loc=0)    
        
    ymu = (x*y.T).T/np.array(mum)

    if c.ndist > 1:
        for k in range(c.ndist):
            #pyp.plot(x,x*y[:,k]/mum[k],'--',label=label)
            ax2.plot(x,ymu[:,k],'--',label=label)
    ax2.plot(x,np.sum(ymu,1),label=label)   
    ax2.legend(loc=0)    
    
    print('Polidispersivity Index estimation:')
    print('mumedio: %s'%(np.sum(N*np.exp(mu)*np.exp(sig**2/2))/sum(N)))
    dmedio  = np.sum(N*np.exp(mu/3)*np.exp(sig**2/18))/sum(N)
    d2medio = np.sum(N*np.exp(2*mu/3)*np.exp(2*sig**2/9))/sum(N)
    print('D-pdi: %s'%(np.sqrt(d2medio/dmedio**2-1)))
    
    return y,x



