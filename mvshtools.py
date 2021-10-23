#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
**********
mvshtools
**********
--------------------------------------
Simple tools for MvsH cycle operation
--------------------------------------

This python-module gives some tools to quantitatively analyze magnetization vs. 
magnetic-applied-field cycles.

    i.   Split conventional two branches loops into two independent branches.
    ii.  Retrieve saturation magnetization using asymptotic behaviour. 
    iii. Calculate (and remove) the superimposed lineal contribution. 
         (:func:`removepara`)
    iv.  Calculate coercivity field.
    v.   Calculate initial susceptibility. 
    vi.  Obtain relevant parameters of Langevin-lik cycle by Chantrell 
         Popplewell and Charles analysis (:func:`cpc`)         

--------------------------------------
cpc method
--------------------------------------
The most used method on this module might be the cpc method. Which
allows obtain <mu> and <mu^2> form near zero and asymptotic behaviour of
Langevin-like magnetic cycle.

Given two arrays H and M with complete cycle, cpc method should 
be called so:

    >>> mt.cpc(H,M,Hmin=6000,rhr=1,limx=50,weight='sep',ob=1,clin=0,T=300);

these kwargs as well others are defined in the method docstring.

Control figures and prints can be disabled making global variable FIGS 
and PRINT False::

    >>> mvshtools.FIGS = False

or::
    
    >>> mvshtools.PRINT = False


Dependencies
-------------
numpy, lmfit, matplotlib


"""
from __future__ import division
from __future__ import print_function

from builtins import object
import matplotlib.pyplot as pyp
import numpy as np
import lmfit


# ------------------------------------------------------------------------------
# I have make a big effort to put all the docstrings
# in English. However, a lot of comments in the code are still in Spanish. 
# ------------------------------------------------------------------------------
__author__      = 'Gustavo Pasquevich'
__version__     = '0.190129'  
_debug          = False
FIGS            = True
PRINT           = True                
__NUMFIG_DEF__  = 9000
NUMFIG          = __NUMFIG_DEF__       # variable for function __newfig__

def splitcycle(H,M):
    """ Split a **M** vs **H** cycle into two branches.

        **H** and **M**  are expect to be close MvsH cycle,  without  
        initial magentisation curve. Both are  lists or numpy.arrays 
        of the same length.

        Returns:
        ========
            
        H1, M1, H2, M2 
    
        where "1" indicate negative  dH/dt branch while "2" indicate the 
	positive dH/dt branch.
    """
    # H1 y M1 is the branch froom H_max to H_min, while H2 and M2 is the branch
    # from H_min to H_max.
     
    if H[0] > H[int(len(H)/2)]:     # Rama inicial decreciente
        _ind_ret = np.where(H == min(H))[0]
        if _debug:
            print(_ind_ret)
        if len(_ind_ret) == 2:
            if (_ind_ret[1]-_ind_ret[0]) == 1:
                i0 = _ind_ret[0]
                i1 = _ind_ret[1]
        else:
            i0 = _ind_ret[0]
            i1 = i0

        if _debug:
            print((i0, i1))

        H1 = H[:i0+1]
        M1 = M[:i0+1]
        H2 = H[i1:]
        M2 = M[i1:]
    else:
        _ind_ret = np.where(H == max(H))[0]
        if len(_ind_ret) == 2 and (_ind_ret[1]-_ind_ret[0]) == 1:
            i0 = _ind_ret[0]
            i1 = _ind_ret[1]
        H2 = H[:i0+1]
        M2 = M[:i0+1]
        H1 = H[i1:]
        M1 = M[i1:]

    return H1,M1,H2,M2

def fitfunc(pars, H, data=None, eps=None):
    """
    Function to be fitted to dataset. Prepared for lmfit module. 

                          (      a       b    )     
    y  =  Ms  * sign(H) * (1 - ----- - -----  ) +  Xi * H   + offset
                          (     |H|     H^2   )   

    This function represents the asymptotic behaviour of a magnetic 
    ferromagnetic component in addition to a lineal contribution, Xi*x.

    """
    # unpack parameters:
    # extract .value attribute for each parameter
    Ms  = pars['Ms'].value
    Xi  = pars['Xi'].value
    ydc = pars['offset'].value
    a   = pars['a'].value
    b   = pars['b'].value

    y = (Ms*np.sign(H))*(1-abs(a)/abs(H)-abs(b)/H**2) + Xi*H + ydc

    if data is None:
        return y
    if eps is None:
        return (y - data)
    return (y - data)/eps



def __fourpoints__(P1,P2,P3,P4):
    """ Given four points P1, P2 P3 and P4 of the M vs H curve this function
        obtains approximate characteristics parameters of the curve::


                             _ ----------     ^
                            - P3       P4     | 
                           /                  | Step
                         _-                   | 
                --------                      |
               P1     P2                      v 


        P1 and P2 should be points in the negative and saturated part of the 
        curve while P3,and P4 in the positive part. This function returns 
        estimators for "step", superimpose linear contribution, and offset.
    """


    x1,y1 = P1
    x2,y2 = P2
    x3,y3 = P3
    x4,y4 = P4

    d3 = (y3 - y1)-(y2 - y1)*(x3 -x1)/(x2-x1)  # d stands for delta ($\delta M$)
    d4 = (y4 - y1)-(y2 - y1)*(x4 -x1)/(x2-x1)
    d1 = (y1 - y4)-(y3 - y4)*(x1 -x4)/(x3-x4)
    d2 = (y2 - y4)-(y3 - y4)*(x2 -x4)/(x3-x4)

    slope12 = (y2 -y1)/(x2-x1)
    slope34 = (y3 -y4)/(x3-x4)

    step = (-d1-d2+d3+d4)/4.
    slope = (slope12 + slope34)/2.
    centro = ((y2-y1)/(x2-x1)*(-x1) + (y3-y4)/(x3-x4)*(-x4) + (y1+y4) )/2.
    
    if FIGS:
        # This figure is a bit strange. It will be plotted in whatewer was 
        # called before this function was called. 
        x = np.array([k[0] for k in [P1,P2,P3,P4]])
        y = np.array([k[1] for k in [P1,P2,P3,P4]])
        pyp.plot(x,y,'o',color='green',alpha=0.5,label='four points')
        pyp.plot([0,0],[-step/2 + centro, step/2+centro],color='k')
        xp = np.array([0,max(x)])
        xm = np.array([min(x),0])
        pyp.plot(xp, step/2 +xp*slope + centro,color = 'k')
        pyp.plot(xm,-step/2 +xm*slope + centro,color = 'k')
        pyp.axhline(0,ls='--',color='k')

        pyp.legend(loc=0)

    return step, slope, centro


def Xi_and_Hc(H1,M1,H2=None,M2=None,limx=188):
    """ Calculates the initial susceptibility and coercive magnetic field.

        Args:
        =====    
        H1,M1 determines the complete cycle or 
              the cycle downward-branch [dH/dt < 0]
        
        H2,M2 determines the cycle rising-branch [dH/dt >0] 
        
        If H2 and M2 are None H1 and M1 are taken as H and M of complet cycle.

        Kwarg:
        ======    
        limx, H-limit to perform the linear fit. It is performed 
              in [-limx,limx]. 
              
       """
    if H2 is None:
        H1,M1,H2,M2 = splitcycle(H1, M1)


    j1 = np.where(np.abs(H1)<limx)[0]
    p1 = np.poly1d(np.polyfit(H1[j1],M1[j1],1))
    j2 = np.where(np.abs(H2)<limx)[0]
    p2 = np.poly1d(np.polyfit(H2[j2],M2[j2],1))

    X = np.mean([p1[1],p2[1]])
    Hc = (-p2[0]/p2[1] + p1[0]/p1[1] ) /2.
    
    if PRINT:
        print('==============================================')
        print('Initial Susceptibility and coercive field') 
        print('----------------------------------------------')
        print(' SLOPE: dM/dH at M=0:                         ')
        print(' [dH/dt<0--branch] :', p1[1])
        print(' [dH/dt>0--branch] :', p2[1])
        print(' mean slope:       :', X)
        print('                                              ')
        print(' Zero croosing magnetic field, H at M=0: ')
        print(' [dH/dt<0--branch]:', -p1[0]/p1[1]) 
        print(' [dH/dt>0--branch]:', -p2[0]/p2[1])
        print(' mean-half-difference: (Hc1-Hc2)/2: Hc= %f Oe'%Hc)
        print('==============================================')

    if FIGS:
        __newfig__(500)
        pyp.cla()
        pyp.plot(H1,M1,'.-r',label='dH/dt<0')
        pyp.plot(H1,p1(H1),color='green')
        pyp.plot(H2,M2,'.-b',label='dH/dt>0')
        pyp.plot(H2,p2(H2),color='green')
        pyp.axvline(-limx,color='k')
        pyp.axvline( limx,color='k')
        pyp.xlim([-188,188])
        pyp.ylim( [ np.min( [ M1[j1].min() , M2[j2].min() ] ) , 
                    np.max( [ M1[j1].max() , M2[j2].max() ] ) ] )
        pyp.axhline(0,color='k')
        pyp.axvline(0,color='k')
        pyp.legend(loc=0)

    return X,Hc

def linealcontribution(H,M,HLIM,E=None,label = 'The M vs H Curve',
                       fixed=dict(),initial=dict(),
                       output=-1):
    """ This function determines the lineal contribution in the M vs H curve, 
    as well as the asymptotic behaviour at high fields. See :func:`fitfunc`.

    Returns
    =======
    lmfit.Parameter instance result of the tail-branch fit of magnetization 
    curve with :func:`fit func`

    =========
    IMPORTANT
    =========
    
    **H** and **M** input arguments corresponds only to **one** of the two 
    MvsH-cycle branches. For a complete cycle it should be called twice, one 
    time for each branch. 

    Arguments
    =========
    args:
    -----    
    1) H, magnetic field. 1d-numpy array.  
    2) M, magnetization.  1d-numpy array.
    3) HLIM = [L1,L2]
        L1 and L2 are two limits defining the fitting zone. The analysis is
        done in the regions [-L1,-L2] and [L2,L1], i.e where  L2 < |H| < L1.

    kwargs:
    -------
    1) E:     weight values for fitting process. Numpy-array same size 
              as **H**. 
    1) label: label for the figures. kwarg for pyplot.plot(). 
              [FIGS global variable should be True]
    2) fixed: dictionary with the parameters that should be fixed:
              e.g. fixed = {'Ms':value,'a':0,'b':None}
                
              valid keys: 'Ms','Xi', 'offset', 'a' and 'b'
              values can be numbers (the values), or None which indicate 
              that the value of the parameters is the automatically obtained 
              but it should be a fix parameter.
    3) initial: dictionary with initial values. If None (or not defined) then 
              they are guessed.
    
    4) output: kind of output, (0 or -1) out.params or (1) out. (v 0.190129)
               Note: output is set to -1 as default to achive  backward 
               compatibility. But it should be changed in future to 1. 
    """

    # Income control. ---------------------------------------------------------
    # --- H,M should be only one branch. (monotonous behaviour test)
    if len(np.unique(np.sign(np.diff(H)))) != 1:
        raise ValueError('Income H parameter should be monotonous (only one branch)')
    # --- H must be a growing list, if the last element is lower than the first, 
    # then the order is inverted.
    if H[-1] < H[0]:    # VERY SENSITIVE ERROR!!!! (before 1/2019 ">")
        H = H[::-1]
        M = M[::-1]

    # E must exist for this routine, if None is created as a list of ones. 
    # Thata is: default = uniform wheight.
    if E is None:
        E = np.ones(H.size)
    # -------------------------------------------------------------------------

    # ======= DEFINNIG INDEXES ------------------------------------------------    
    # H_LIM_1 and 2 are whish limits. L1a, L2a, L2b and L1b are index in H that
    # satisfies this condition, going from lower H to highre H. j are the index
    # where H_LIM_2<|H|<H_LIM_1.
    
    H_LIM_1, H_LIM_2 = HLIM 

    # L1 es el limite de alto campo. L1a corresponde a la region de campos 
    # positivos y L1b a la de negativos.
    if H_LIM_1 >= max(abs(H)):
        L1a = 0
        L1b = len(H)-1
    else:
        L1a = np.where( H >= -H_LIM_1)[0][0]
        L1b = np.where( H <= H_LIM_1)[0][-1]        

    # L2 is the low-field limit. L2a correspond to positive fields while L1a
    # to negative ones.
    L2a = np.where( H <= -H_LIM_2)[0][-1]
    L2b = np.where( H >= H_LIM_2)[0][0]     

    j = np.where( (np.abs(H) <= H_LIM_1) & (np.abs(H) >= H_LIM_2) )[0]
    # --- END INDEXES DEFINITION ---------------------------------------------- 

    # Estimation of initial parameters
    if FIGS:
        # This figure is used as a previus figure definition to be used by 
        # __four__points function. 
        __newfig__()
        pyp.plot(H,M,'.-',color='gray')
        pyp.title('4points for initial parameters guessing')
    P1,P2,P3,P4 = [[H[k],M[k]] for k in [L1a,L2a,L2b,L1b]]
    salto, pendiente, centro  = __fourpoints__(P1,P2,P3,P4)
        

    h_fit = H[j]
    m_fit = M[j]
    e_fit = E[j]
    # INITIATING LMFIT ------------------------------------
    # 
    # Building parameteters dictionary. lmfit.Parameters
    params = lmfit.Parameters()
    params.add_many(('Ms',salto/2., True, None,None,None),
                    ('Xi',pendiente,True, None,None,None),
                    ('offset',centro,True, None,None,None),
                    ('a',0.0079,True, None,None,None),
                    ('b',0.0001,True, None,None,None))

    # Se fijan y definen los parametros segun las entrdas *fixed* y *initial*
    for k in list(initial.keys()):
        params[k].value = initial[k]
    for k in list(fixed.keys()):
        params[k].vary = False
        if fixed[k] != None:
            params[k].value = fixed[k]
    
    out = lmfit.minimize(fitfunc, params, args=(h_fit,m_fit,e_fit), kws=None, 
                         method='leastsq')
    out.dataX = h_fit
    out.dataY = m_fit
    out.dataE = e_fit

    if PRINT:
        print(PRINT)
        print('Information on the pre fit and pos fit parameters')
        print('-------------------------------------------------')
        for k in list(params.values()):
            print(k) 
        lmfit.report_fit(out.params) # print fit report

    if FIGS:
        __newfig__()
        ax = pyp.gca()
        ax.axvspan(H_LIM_1,H_LIM_2,color='yellow',alpha=0.5)
        ax.axvspan(-H_LIM_1,-H_LIM_2,color='yellow',alpha=0.5)

        pyp.plot(H,M,'.')
        pyp.plot(H,fitfunc(params,H),'-r',lw=2,alpha=0.8,label='pre fit curve')
        pyp.plot(h_fit,m_fit,'x',label='selected data to be fitted',ms=10.)
        pyp.plot(H,fitfunc(out.params,H),color='k',lw=1,ls='--',label='fitted curve')
        pyp.title(label)
        pyp.ylim([M.min(),M.max()])
        pyp.ylim([M.min(),M.max()])
        x = [k[0] for k in [P1,P2,P3,P4]]
        y = [k[1] for k in [P1,P2,P3,P4]]
        pyp.plot(x,y,'o',color='green',alpha=0.5,label='four points')
        pyp.legend(loc=0)
        
     
    # All this strange output is to make soft version tranistion between default
    # output 0 and default otput 1.     
    if output == 1:
        return out
    else:
        if output ==-1:
            print('***** In future linealcontribution will return as defult out instead out.params****')
        return out.params



def removepara(H,M,Hmin = '1/2',Hmax = 'max',output=-1,kwlc={}):
    """ Retrieve lineal contribution to cycle and remove it from cycle.


        **H** y **M** corresponds to entire cycle (two branches). I.e. **H** 
        starts and ends at the same value (or an aproximate value).

        El ciclo M vs H se separa en sus dos ramas. H1,M1 y H2,M2, defined by:: 

            H1,M1: curva con dH/dt < 0. El campo decrece con el tiempo.
            H2,M2: curva con dH/dt > 0. El campo aumenta con el tiempo.

        Con la variable global FIGS = True shows intermediate states of  
        proceso de determinarion y linear contribution removing.
        Figure Shows **Hmin** and **Hmax** positions in the cycle.
        
        output: kind of output, (0 or -1) out.params or (1) out. (v 0.210304)
               Note: output is set to -1 as default to achive  backward 
               compatibility. But it should be changed in future to 1.
        
        kwlc = dictionary with kwargs to be passed to lienar contribution. 

        Returns: 
            if output = -1: H1,M1,H2,M2,[pendiente,salto,desp]
            if output = 1: 
                returns plain objtect with previous attributes and others. 
                                            
    """
    if PRINT:
        print('**********************************************************')
        print('removepara ')
        print('**********************************************************')

    if Hmax == 'max':
        Hmax = max(abs(H))
    if Hmin == '1/2':
        Hmin = 0.5*max(abs(H))

    H1,M1,H2,M2 = splitcycle(H,M)

    o1 = linealcontribution(H1,M1,[Hmax,Hmin],label='dH/dt < 0',output=output,**kwlc)
    o2 = linealcontribution(H2,M2,[Hmax,Hmin],label='dH/dt > 0',output=output,**kwlc)
    if output == 1:
        p1 = o1.params
        p2 = o2.params
    elif output == -1:
        p1 = o1
        p2 = o2
        
    Ms = (p1['Ms'].value + p2['Ms'].value)*0.5
    if p1['Ms'].stderr == None or p2['Ms'].stderr == None:
        eMs = None
    else: 
        eMs = (p1['Ms'].stderr + p2['Ms'].stderr)*0.5

    # Fin de ajustes 
    if PRINT:
        print('slope     1:',p1['Xi'])
        print('slope     2:',p2['Xi'])
        print('Ms 1       :',p1['Ms'])
        print('Ms 2       :',p2['Ms'])
        print('Ms         :%s +/- %s'%(Ms,eMs))
        print('offset  1  :',p1['offset'])
        print('offset  2  :',p2['offset'])
        print('a  1       :',p1['a'])
        print('a  2       :',p2['a'])
        print('b  1       :',p1['b'])
        print('b  2       :',p2['b'])
        


    # Armamos una pendiente promedio a partir de la obtenida para cada rama.
    # Corregimos ambas ramas eliminando esta pendiente.

    pend =(p1['Xi']+p2['Xi'])/2.
    salto=(p1['Ms']+p2['Ms'])/2.
    desp =(p1['offset']+p2['offset'])/2.
    M1 = (M1-H1*pend)
    M2 = (M2-H2*pend)

    if FIGS:
        __newfig__()
        pyp.plot(H1,M1,'b.-',label = 'dH/dt < 0')
        pyp.plot(H2,M2,'r.-',label = 'dH/dt > 0')
        pyp.axhline(salto,color = 'k', alpha =0.5)
        pyp.axhline(-salto,color= 'k', alpha =0.5)
        pyp.legend(loc=0)
        
    if output == 1:
        out = ReturnClass()
        out.H1 = H1
        out.H2 = H2
        out.M1 = M1
        out.M2 = M2
        out.pend = pend
        out.desp = desp
        out.salto = salto
        out.o1 = o1
        out.o2 = o2
        return out
    else:
        return H1,M1,H2,M2,[pend,salto,desp]


def cpc(H, M, Hmin = '1/2', Hmax = 'max', clin=None, T=300, limx=10, 
              weight='None', rhr=False, aini=100., ob = False,
              __outpk__=-1):
    """ Given a superparamagnetic cycle M vs H, assuming distribution of Lagevin 
        functions calculate <mu>, N and <mu^2> from M vs H cycle. 

        It perform similar analysis as proposed by R. Chantrell, J. Popplewell 
        and S. Charles in "Measurements of particle size distribution parameters 
        in ferrofluids"  IEEE Transactions on Magnetics ( Volume: 14, Issue: 5, 
        September 1978). pag. 975 - 977 DOI:10.1109/TMAG.1978.1059918.  

        The relevant information of analysis is printed on screen. See below 
        `printed information`.

        Arguments:
        ==========
        H and M: 
                They are assumed to be in **cgs units**. H in Oe and M in 
                emu/sth. **sth** stands for something, indicating the fact that
                the moment or magnetisation values can be given by grams, cm^3,
                whatever or even nothing.
                H and M are np.arrays that with a whole cycle, i.e they include
                both branches. In case it is only one branch the kwarg **ob** 
                must be set True.

        kwargs:
        =======
        Hmin,Hmax: 
            They define the high field region for the asymptotic analysis. 
            The high field region is defined so that absolute value of H is 
            between **Hmin** and **Hmax**.
                #. **Hmin** can be a value or the string '1/2' (default), which 
                    means is taken as the half of the maximum values of |H|.
                #. **Hmax** can be a value or the string 'max' (default) which 
                    indicates that Hmax must be taken as the maximum of |H|.

        clin:   
            value of lineal-constant to be take as a fixed parameter.
            (if None [default], then is fitted) 

        a:      
            proposing initial value for "a" parameter in asymptotic process.
 
        T:      
            Temperature used for Chanterell calculations,
        limx:  
            max H field for calculation dM/dH at H=0.
        weight: 
            String to indicates the fitting weight for the asymptotics 
            behaviour fit. It is relevant if there absice points are not 
            uniformly separated.
                'None': 
                    uniform wheight.
                'sep' : 
                    inverse proportional to separation between points H-value.
        rhr:    
            remove-H-remanenet. Before the analysis, it shift the values of H 
            so they not have remanent field. That is acceptable if measurement 
            provides from a SQUID magnetometer, and if it's known that the 
            sample behaves does not have coercive field. However here is only 
            a technical parameter to better determine suceptibility at low 
            fields.

        ob:     
            {False} or True. ob = one-branch. Is true if in H and M vectors 
            correspond only to a branch. 

        Returns: 
        ========
            -- H1, M1, H2, M2, [slop,step,offset], mu, N, mumu (if 
                   __outpk__ -1 or 0).            
            -- plain instance with previous atributes (if __outptk__ is 1):
               out.H1:      H1
               out.H2:      H2
               out.M1:      M1
               out.M2:      M2
               out.slope:   high-field-lineal-contribution slope.
               out.step :   saturation magntization (or moment or whatever: Ms)
               out.offset:  offset
               out.mu:      mean magentic moment (number distribution)
               out.N:       number of particles (or whatever)
               out.mumu:    mean magnetic momente (moment distribution)
               out.o1:      output fo lmfit minimizer corresponding to first branch.
               out.o2:      output of lmfit minimizer corresponding to second branch.
               

        Printed Information:
        ====================
        1: <mu> = magnetic moment retrieved from 1st branch. In $\mu_B$
        2: <mu> = magnetic moment retrieved from 2nd branch. In $\mu_B$
        1: N    = number per particles per something (depends on the units 
                  of **M**). retrieved from 1st branch. 
        2: N    = idem but retrieved from 2nd branch. 

        <mu>    = magnetic moment (mean value of 1st and 2nd branches results) 
                                   and mean difference for uncertainty).   
        sqrt(<mu^2>) = (what it means)  
        N       = mean value of N obtained from both branches. 
        rho     = <mu>^2/<mu^2>  (as defined by Allia et al.)
        STD     = standar deviation of moments distribution sqrt( <mu^2>-<mu>^2 )
        sigma-lognormal = 
                  sigma parameter of a lognormal with same mean value and 
                  standar-deviation 
        <mu>_mu = mean magnetic moment according moment-distribution. 


        **M** vs **H** cycle is splitted into their two branches,
        H1,M1 y H2,M2, defined by:: 

            H1,M1: curva con dH/dt < 0. El campo decrece con el tiempo.
            H2,M2: curva con dH/dt > 0. El campo aumenta con el tiempo.

        If global variable is set true control graphics are plotted during
        analysis. Is recommended to use graphics to control the goodness of 
        analysis.   

            >>> import mvshtools as mt
            >>> mt.FIGS = True


        The function has been called first as the first author of the paper
        in which is base on. However, I thought it wasn't correct name only one 
        of the names of a three authors of the paper. So the functon was 
        then named with the first letter of each one of the authors: **cpc**. 
               
    """

    """ kwargument __outpk__ (output kind) changes output variable from 
        older version to newer. Old way was a list, now it a a plain instance 
        with attributes.
    """ 

    __newfig__(reset = True)
    
    # Physics Constants
    kB  = 1.3806488e-16  #erg/K Boltzmann Constant in cgs units
    muB = 9.27400968e-21 #erg/G Bohr Magneton in cgs units.

    # Very strange method to resolve situation were there is only a branch. It
    # should be improved
    if ob:  
        H1,H2,M1,M2 = H, H, M ,M
    else:    
        H1,M1,H2,M2 = splitcycle(H,M)
    
    if rhr:
        H1,r1 = remove_H_remanent(H1,M1,outmore=True)
        H2,r2 = remove_H_remanent(H2,M2,outmore=True)
        Hc_rhr= (r2.Hr - r1.Hr)/2

    H1max = max(np.abs(H1))
    H2max = max(np.abs(H2))

    if Hmax == 'max':
        Hmax = max(H1max,H2max)
    if Hmin == '1/2':
        Hmin = 0.5*max(H1max,H2max)

     
    # Low magnetic field analysis===============================================    
    X,HC = Xi_and_Hc(H1,M1,H2,M2,limx=limx)

    # ==========================================================================
    #Tail zone analysis of tail zone 

    if weight == 'sep':
        E1 = __makeEdiff__(H1)
        E2 = __makeEdiff__(H2)
    else:
        E1 = None
        E2 = None

    fixed = dict()
    initial=dict()
    initial['a']=aini
    if clin is not None:
        fixed['Xi'] = clin
    fixed['b']  = 1e-10
    
    if PRINT: 
        print('\n\nWorking whit dH/dt < 0 branch \n-----------------------------\n')
    o1 = linealcontribution(H1, M1, [Hmax,Hmin], label='dH/dt < 0', fixed=fixed, 
                            E=E1, initial = initial,output=1)
    p1 = o1.params

    if PRINT: 
        print('Reduce ChiSqr',o1.redchi)
    
    if PRINT: 
        print('\n\nWorking whit dH/dt > 0 branch \n-----------------------------\n')
    o2 = linealcontribution(H2,M2,[Hmax,Hmin],label='dH/dt > 0',fixed=fixed,
                            E=E2,initial=initial,output=1)
    p2 = o2.params

    if PRINT: 
        print('Reduce ChiSqr',o2.redchi)


    pend  = (p1['Xi']     + p2['Xi'])/2.
    salto = (p1['Ms']     + p2['Ms'])/2.
    desp  = (p1['offset'] + p2['offset'])/2.
    M1 = (M1-H1*pend)
    M2 = (M2-H2*pend)




    # Finish wit cycle analysis 
    # ========================================================================= 
    if PRINT:
        print('========================')
        print('cycle-analysis results')
        print('========================')
        print('slope 1    :'    ,p1['Xi'].value)
        print('slope 2    :'    ,p2['Xi'].value)
        print('step 1     :' ,p1['Ms'].value)
        print('step 2     :' ,p2['Ms'].value)
        print('offset  1  :',p1['offset'].value)
        print('offset  2  :',p2['offset'].value)
        print('a  1       :',p1['a'].value)
        print('a  2       :',p2['a'].value)
        print('b  1       :',p1['b'].value)
        print('b  2       :',p2['b'].value)
        print('HC         :',HC)
        print('X          :',X)


    desp     = (p1['offset'].value + p2['offset'].value)/2.
    
    mu1      = kB*T/p1['a']/muB
    mu2      = kB*T/p2['a']/muB
    emu      = abs(mu1-mu2)/2. 
    mu       = (mu1+mu2)/2.
    N1       = p1['Ms']/mu1/muB
    N2       = p2['Ms']/mu2/muB
    N        = (N1+N2)/2.
    mumu     = (X-pend)*3*kB*T/N/muB**2 
    
    rhoA     = mumu/mu**2
    STD      = np.sqrt(mumu-mu**2)
    sigma    = np.sqrt(np.log(rhoA))

    if PRINT:
        if rhr:
            print('Hc_rhr  =  %.1f'%Hc_rhr)
        print('----------------------------------')
        print('1: <mu> = %.1f muB'%(mu1))
        print('2: <mu> = %.1f muB'%(mu2))
        print('1: N    = %.2e'%(N1))
        print('2: N    = %.2e'%(N2))
        print('----------------------------------')
        print('<mu> .......... = %.1f mB +/- %.1f'%(mu,emu)) 
        print('sqrt(<mu^2>)... = %.1f mB'%(np.sqrt(mumu)))
        print('N ............. = %.2e num/sth.'%(N))
        print('rho ........... = %.2f '%(rhoA))
        print('STD ........... = %.1f mB'%(STD))
        print('sigma-lognormal = %.2f '%(sigma))
        print('----------------------------------')
        print('<mu>_mu ....... = %.1f mB'%(mumu/mu))


    if FIGS:
        __newfig__()
        pyp.plot(H1,M1-desp,'b.-',label = 'dH/dt < 0')
        pyp.plot(H2,M2-desp,'r.-',label = 'dH/dt > 0')
        pyp.axhline(salto,color = 'k', alpha =0.5)
        pyp.axhline(-salto,color= 'k', alpha =0.5)
        pyp.legend(loc=0)

    if __outpk__ == 1:
        ret = ReturnClass()
        ret.H1 = H1
        ret.H2 = H2
        ret.M1 = M1
        ret.M2 = M2
        ret.slope = pend
        ret.step  = salto
        ret.offset = desp
        ret.mu = mu
        ret.N  = N
        ret.mumu = mumu
        ret.o1 = o1
        ret.o2 = o2
        return ret
    else:
        if __outpk__ == -1:
            print('***** In future cpc will return as defult an empty-class instance ' \
                  'with attributes instead a list. ****')
        return H1,M1,H2,M2,[pend,salto,desp],mu,N,mumu

def remove_H_remanent(H,M,limx=None,outmore = False):
    """ Removes the remanent applied magnetic field that sometimes appear in 
        superconductors SQUID magnetometers. 
        
        In fact this function removes coercive field by shiftting and 
        assuring zero final coercive field. This is to say, removes coercive 
        field by by simple **Hc** subtraction. This function 
        should be used only if correspond.

        Arguments:
        ----------
        **H** and **M** should be an unique branch. 
        From **M** extract **Hc** and returns **H-Hc**.
       
        kwargs:
        ------
        **limx**: region (|**H**| < **limx**) to look for the coercive field. 
        If not given it use 10% of maximum abs(**H**). **limx** must be 
        choosen so **M** is linearly dependent with **H** inside those limits. 
 
        **outmore**, if True (default is False) returns as second argument more
        information.
       
        Returns:
        --------
        Returns **H** - **Hc**

        If **outmore** returns structure with **Hr** attribute. 

    """
    
    if limx is None:
        limx = max(np.abs(H))*.1
    j1 = np.where(np.abs(H)<limx)[0]
    
    def Hr(H,M):
        if np.mean(np.diff(H)<0):
            Hr = np.interp(0,M[j1][::-1],H[j1][::-1])
        elif np.mean(np.diff(H)>0):
            Hr = np.interp(0,M[j1],H[j1])
        return Hr
    Hr = Hr(H,M)
    
    H = H-Hr
 
    if outmore:
        r = ReturnClass()
        r.Hr = Hr
        return H, r
    else:
        return H

def anhysteretic(H,M):
    """ Calculate anhysteretic curve as proposed by Allia at al.  """
    H1,M1,H2,M2 = splitcycle(H,M)
    H = (H2+H1[::-1])/2
    M1i = np.interp(H,H1[::-1],M1[::-1])
    M2i = np.interp(H,H2,M2)
    M = (M1i+M2i)/2

    if FIGS:
        __newfig__()
        pyp.plot(H1,M1,'b.-',alpha =0.4,label = 'dH/dt < 0')
        pyp.plot(H2,M2,'r.-',alpha =0.4, label = 'dH/dt > 0')
        pyp.plot(H,M,'k.-',label = 'Anhysteretic ')
        pyp.legend(loc=0)
    return H,M
    


def cyclewithoutHc(H,M):
    """ Returns a similar cycle without corecive field (by horizontal shift)"""
    H1,M1,H2,M2 = splitcycle(H,M)
    H1 = remove_H_remanent(H1,M1)
    H2 = remove_H_remanent(H2,M2)
    H = np.concatenate([H1,H2])
    M = np.concatenate([M1,M2])
    return H,M


def __makeEdiff__(H,smallvalue=1e-10 ):
    """ Calculates the inverse of the mean difference of each point with their
        neighbours. All points are assigned to a value given by their two 
        neighbours, except the extreme points that only have one neighbour. 
        
        To avoid division by zero all values are incremented in **smallvalue**. 
        Default value should work without problem. 
        (comment: division by zero would have happened if two suscesive points 
        are measured at the same "x-value") 
    """
    D = np.diff(H)        
    D1 = np.append(D[0],D)
    D2 = np.append(D,D[-1])
    D = (D1+D2)/2.+smallvalue 
    return 1/D


def __newfig__(num=None,reset=False):
    """ Auxiliary function to define  new-figures id-numbers.

        kwargs
        ------
        reset: 
            for initialize numebering in default value. Entering by 
            this mode, only NUMFIG is reseting. It doesen't create a 
            new figure. 
        num: 
            skip the fundamental idea of __newfig__, and works exactly 
            as figure. 
    """
    global NUMFIG

    if reset:
        NUMFIG = __NUMFIG_DEF__
        return
    
    if num == None:
        pyp.figure(NUMFIG)
        pyp.cla()
        NUMFIG += 1
    else:
        pyp.figure(num)
        pyp.cla()

    return NUMFIG

class ReturnClass(object):
    pass


