"""
Monteary utilities
-------------------
Closely linked to deep hedging http://deephedging.com/
March 1st, 2023
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config
from .util import tf
_log = Logger(__file__)

# -------------------------------------------------
# utilities for managing layers
# -------------------------------------------------

UTILITIES = ['mean', 'exp', 'exp2', 'vicky', 'cvar', 'quad']

@tf.function  
def tf_utility( utility : str, lmbda : float, X : tf.Tensor, y : tf.Tensor = 0., return_derivative : bool = False ) -> dict:
    """
    Computes
        u(X+y) - y
    and its derivative in X for random variable X and OCE variable y.
    Read the material on http://deephedging.com/ for more details.

    Parameters
    ----------
    utility: str
        Which utility function 'u' to use
    lmbda : flost
        risk aversion
    X: tf.Tensor
        Random variable, typically total gains on the path
    y: tf.Tensor, None, or 0
        OCE intercept y.

    Returns
    -------
        dict if return_derivative is True
            with menbers 'u' and 'd'
        the utility value if return_derivative is False
    """
    utility  = str(utility)
    lmbda    = float(lmbda)
    y        = y if not y is None else 0.
    gains    = X + y
    
    _log.verify( lmbda >= 0., "Risk aversion 'lmbda' cannot be negative. Found %g", lmbda )
    if lmbda < 1E-12: 
        # Zero lambda => mean
        utility = "mean"
        lmbda   = 1.

    if utility in ["mean", "expectation"]:
        # Expectation
        #
        u = gains
        d = tf.ones_like(gains)
        
    elif utility == "cvar":
        # CVaR risk measure
        #   u(x) = (1+lambda) min(0, x)
        # The resulting OCE measure U computes the expected value under the condition that X is below the p's percentile.
        #   U(X) = E[ X | X <= P^{-1}[ X<=* ](p)
        # Here p is small to reflect risk-aversion, e.g p=5% means we are computing the mean over the five worst percentiles.
        # Note that CVaR is often quoted with a survival percentile, e.g. q = 1-p e.g. 95%
        #
        # Conversion from percentile p (e.g. 5%) 
        #   1+lambda = 1/p 
        # =>
        #   lambda = 1/p - 1
        #
        # Conversion from lambda to percentile
        #   p = 1/(1+lambda)
        #
        # In other words, for p=50% use 1. (as in https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3120710)
        #                 for p=5%  use 19.
        
        u = (1.+lmbda) * tf.math.minimum( 0., gains ) - y
        d = tf.where( gains < 0., -(1.+lmbda), 0. )

    elif utility == "quad":
        # quadratic penalty; flat extrapolation
        #
        # u(x)  = -0.5 lambda * ( x - x0 )^2 + 0.5 * lmbda * x0^2;   u(0)  = 0
        # u'(x) = - lambda (x-x0);                                   u'(0) = 1 = lambda x0 => x0 = 1/lmbda            
        
        x0 = 1./lmbda
        xx = tf.minimum( 0., gains-x0 )
        u  = - 0.5 * lmbda * (xx**2) + 0.5 * lmbda * (x0**2) - y
        d  = - lmbda * xx
                
    elif utility in ["exp", "entropy"]: 
        # Entropy
        #   u(x) = { 1 - exp(- lambda x ) } / lambda 
        #
        # The OCE measure for this utility has the closed form
        #   U(X) = - 1/lambda log E[ exp(-\lambda X) ]
        #
        # However, this tends to be numerically challenging.
        # we introcue a robust version less likely to explode
        inf = tf.stop_gradient( tf.reduce_min( X ) )
        u = (1. - tf.math.exp( - lmbda * (gains-inf)) ) / lmbda - y + inf
        d = tf.math.exp(- lmbda * gains )
        
    elif utility == "exp2":
        # Exponential for the positive axis, quadratic for the negative axis.
        # A non-exploding version of the entropy
        #
        # u1(x)  = { 1-exp(-lambda x) } / lambda; u1(0)  = 0 
        # u1'(x) = exp(-lambda x);                u1'(0) = 1       
        # u2(x)  = x - 0.5 lambda x^2;            u2(0)  = 0
        # u2'(x) = 1 - lambda x;                  u2'(0) = 1
        g1  = tf.maximum(gains,0.)
        g2  = tf.minimum(gains,0.)
        eg1 = tf.math.exp( - lmbda * g1)
        u1  = (1. - eg1 ) / lmbda - y            
        u2  = g2 - 0.5 * lmbda * g2 * g2 - y
        d1  = eg1
        d2  = 1. - lmbda * g2
        u   = tf.where( gains > 0., u1, u2 )
        d   = tf.where( gains > 0., d1, d2 )
        
    elif utility == "vicky":
        # Vicky Handerson & Mark Rodgers
        # https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/henderson/publications/indifference_survey.pdf
        #
        # u(x)  = { 1 + lambda * x - sqrt{ 1+lambda^2*x^2 } } / lmbda
        # u'(x) = 1 - lambda x / sqrt{1+lambda^2*x^2}
        u = (1. + lmbda * gains - tf.math.sqrt( 1. + (lmbda * gains) ** 2 )) / lmbda  - y
        d = 1 - lmbda * gains / tf.math.sqrt( 1. + (lmbda * gains) ** 2)
        
    _log.verify( not u is None, "Unknown utility function '%s'. Use one of %s", utility, fmt_list( MonetaryUtility.UTILITIES )  )
    
    u = tf.debugging.check_numerics(u, "Numerical error computing u in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    d = tf.debugging.check_numerics(d, "Numerical error computing d in %s. Turn on tf.enable_check_numerics to find the root cause.\nX: %s\ny : %s" % (__file__, str(X), str(y)) )
    
    return pdct(
            u = u,
            d = d
        ) if return_derivative else u