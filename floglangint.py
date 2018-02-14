#!/usr/bin/env python
# coding: utf8

""" Fast Lognormal-Langevin Integration. (floglangint)

This module perform the integral of product of Langevin function and lognormal 
function, acording as is used in magentic issues. The integral is:

.. math::

  \Int_0^{\infty} x Lan(\\alpha x ) f(x;\mu,\sigma) dx

where :math:`Lan(x)` is the Langevin function:

.. math::

    Lan(x) = \coth(x)- 1/x

and :math:`f(x;\mu,\sigma)` the LogNormal distribution::

.. math::

    f(x;\mu,\sigma) = \\frac{1}{(\sqrt(2*\pi) \sigma x} e^{-(\ln(x)-\mu)^2/2\sigma^2 } 

The numeric aproximation has a absolute error lower than 1e-5.

The function is called as:: 

    I = integral(alpha,mu,sigma)

where,
    alpha = factor multipling to the distributed variable x in the Langevin 
            function.
    mu = LogNormal mu parameter
    sigma = LogNormal sigma parameter

Examples:
In [4]: integral(1,7.,1.)
Out[4]: 1807.0424144562169

In [5]: integral(2.,7.,1.)
Out[5]: 1807.5424144560643

In [6]: integral(2.,10.,1.)
Out[6]: 36315.002674246636


"""

# developing directory:
# /home/gustavo/Atlantis/FISICA/EXPERIMENTOS/2015_int_num_lognorm_lang

import numpy as np
from scipy.stats import norm

__version__ = '2017.1030'
__author__ = 'Gustavo A. Pasquevich'


# ====================================================
# result of:  w13.fig100(n=10,a=1,b=6.4,ep=0,pp=1)
# ====================================================
c2 = [-4.1725712798497535917571906946554882722466572886332869529724121093750000e-08,
1.5486466480833479401245595177827318877916695782914757728576660156250000e-06,
-2.4201283750948089466955778781276364952645963057875633239746093750000000e-05,
2.0016015746101556145412903031655105223762802779674530029296875000000000e-04,
-8.4801154069656886711620069618788875231985002756118774414062500000000000e-04,
7.1366383712397421252432438976143203035462647676467895507812500000000000e-04,
1.0135491220109158275186622688579518580809235572814941406250000000000000e-02,
-4.4324480130278121059461682307301089167594909667968750000000000000000000e-02,
2.5198208898557290791320184553114813752472400665283203125000000000000000e-02,
3.1822533969645588891594911729043815284967422485351562500000000000000000e-01,
3.7588899913298945421047392301261425018310546875000000000000000000000000e-03]
c2.reverse()

c1= [1/3.,-2.2222222222222e-02,2.11640211640212044e-03,-2.116402116402113238024e-04]
 

# Auxiliares ===================================================================
def __muk__(mu,sigma,k):
    """ k-esimo momento de la distribución Lognormal de parámetros mu y sigma.
        mu_k = < x^k >  """
    return np.exp(k*mu+0.5*k**2*sigma**2)

def __Ia_k__(mu,sigma,a,k):
    """ Integral of x^k*lognorm(x,mu,sigma) between  "0" and "a" """
    #print mu,sigma,a,k,muk(mu,sigma,k),norm.cdf( (np.log(a)-mu-k*sigma**2)/sigma  )
    return __muk__(mu,sigma,k)*norm.cdf( (np.log(a)-mu-k*sigma**2)/sigma  ) 

def __Iab_k__(mu,sigma,a,b,k):
    """ Integral of x^k*lognorm(x;mu,sigma) between "a" and "b". """
    return __Ia_k__(mu,sigma,b,k) - __Ia_k__(mu,sigma,a,k)

def __Iainf_k__(mu,sigma,a,k):
    """ Integral of x^k*lognorm(x,mu,sigma) between "a" and Infinity. """
    return __muk__(mu,sigma,k)*norm.cdf( (mu + k*sigma**2-np.log(a))/sigma  ) 

# ==============================================================================
# Aproximation of integral in three regions ====================================
def __Ilow__(alpha,mu,sigma):
    """ Integral in [0, 1/alpha]"""
    b = 1.
    I = np.zeros(alpha.shape)
    for i, c in enumerate(c1):
        I += c*alpha**(2*i+1)*__Ia_k__(mu,sigma,b/alpha,2*i+2)
    return I

def __Imed__(alpha,mu,sigma):
    """ Integral in [1./alpha, 6.4/alpha]"""
    a,b = 1., 6.4
    I = np.zeros(alpha.shape)
    for i, c in enumerate(c2):
        I += c*alpha**i*__Iab_k__(mu,sigma,a/alpha,b/alpha,i+1)
    return I

def __Ihigh__(alpha,mu,sigma):
    """ Integral in [6.4/alpha, infinity]"""
    b = 6.4
    return __Iainf_k__(mu,sigma,b/alpha,1)-1/alpha*__Iainf_k__(mu,sigma,b/alpha,0)


# ==============================================================================
def integral(alpha,mu,sigma,split=False):
    """ Integral of x*Langevin(alpha*x)*Logorm(x;mu,sigma) between 0 and Infinity. 
        alpha: numpy.array"""
    sign = np.sign(alpha)
    alpha = np.abs(alpha)
    

    I1 = sign*__Ilow__(alpha,mu,sigma) 
    I2 = sign*__Imed__(alpha,mu,sigma)
    I3 = sign*__Ihigh__(alpha,mu,sigma)

    if not split:
        return I1+I2+I3
    else:
        return I1+I2+I3,I1,I2,I3 

# ------------------------------------------------------------------------------
# Historial de cambios al pie del módulo 
# ------------------------------------------------------------------------------
# v.160401: 
#   Agrego la posibilidad de llamar la integral con valores negativos. 
# v.2016.0519: 
#   Mejoro el docstring inicial.
# v.2017.0316: 
#   Hay una sentencia en el primer renglón del docstring que me confunde
#   y obviamente debe etar mal. Decía:
#       Integrate 1/x Lan(alpha x ) f(x;mu,sigma) dx between 0 and ininity.
#   Pero ese x dividiendo no se condice con lo que queríamos calcular ni con 
#   lo que dice el docstring de la función "integral". El código es difícil de 
#   seguir, habría que documentarlo mejor. Pero por ahora solo infomo que cambié
#   el docstring y entendemos que la integral es x Lan(x*a)*f(x,mu,sig)
# v.2017.1030:
#   Cambio el docstring para que los caractéres \a y \f se lean correctos.
#
