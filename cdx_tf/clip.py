"""
Various methods for clipping
-----------------------------
March 1st, 2023
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config
from .util import tf, def_dtype
_log = Logger(__file__)

@tf.function
def hard_clip( x : tf.Tensor, x_min : tf.Tensor, x_max : tf.Tensor, validate_bounds : bool = False ) -> tf.Tensor:
    """
    Hard clips x between 'x_max' and 'x_min' with quadratic penalty on exceeding the bounds.
    The functiuon will compute y min( max_x, max( min_x )) and return (y, abs(x-y)**2 )
    
    Parameters
    ----------
        x : tf.Tensor
        x_max, x_min: tf.Tensor
            Bounds
        validate_bounds : if True, check that x_min<x_max
        
    Returns
    -------
        Returns a tuple x, and p:
            x : truncated value for x
            p : penalty of the form (x-x_new)**2 which should be added to the overall training objective
                to ensure there is a gradient towards the interior of the clip
    
    """
    if not validate_bounds:
        x_ = tf.where( maxx > minx, tf.clip_by_value( x, minx, maxx ), 0.5*(maxx+minx) )
        p  = tf.math.abs( x - x_ ) ** 2
        return x_, p
    with tf.control_dependencies( [ tf.debugging.assert_greate( maxx, minx, message="Upper bound for trades must be bigger than lower bound" ) ] ):
        return hard_clip( x, x_min, x_max, validate_bounds = False )
    
try:
    import tensorflow_probability as tfp
except:
    tfp = None 
   
class SoftClip(tf.keras.layers.Layer):
    """
    Simple wrapper around tensorflow_probability.bijectors.SoftClip with a number of features
    Additional config variables are
        hard_clip: hard clip, no soft clipping (this is intended for debugging)
        outer_clip: appy outer clip, e.g. hard clip beyond an excessive rannge
        outer_clip_cut_off: bounds for the outer clip
    """

    def __init__(self, config, name : str = None, dtype : tf.DType = def_dtype ):
        """
        Initialize softclip from tensorflow_probability
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )   
        _log.verify( not tfp is None, "tensorflow_probability package not found")      
        
        self.hard_clip             = config('hard_clip', False, bool, "Use min/max instread of soft clip for limiting actions by their bounds")
        self.outer_clip            = config('outer_clip', True, bool, "Apply a hard clip 'outer_clip_cut_off' times the boundaries")
        self.outer_clip_cut_off    = config('outer_clip_cut_off', 10., Float>=1., "Multiplier on bounds for outer_clip")
        hinge_softness             = config('softclip_hinge_softness', 1., Float>0., "Specifies softness of bounding actions between lbnd_a and ubnd_a")
        self.softclip              = tfp.bijectors.SoftClip( low=0., high=1., hinge_softness=hinge_softness, name='soft_clip' if name is None else name )
        config.done()
    
    def __call__( self, actions, lbnd_a, ubnd_a ):
        """ Clip the action within lbnd_a, ubnd_a """
        
        with tf.control_dependencies( [ tf.debugging.assert_greater_equal( ubnd_a, lbnd_a, message="Upper bound for actions must be bigger than lower bound" ),
                                        tf.debugging.assert_greater_equal( ubnd_a, 0., message="Upper bound for actions must not be negative" ),
                                        tf.debugging.assert_less_equal( lbnd_a, 0., message="Lower bound for actions must not be positive" ) ] ):
        
            if self.hard_clip:
                # hard clip
                # this is recommended for debugging only.
                # soft clipping should lead to smoother gradients
                actions = tf.minimum( actions, ubnd_a, name="hard_clip_min" )
                actions = tf.maximum( actions, lbnd_a, name="hard_clip_max" )
                return actions            

            if self.outer_clip:
                # to avoid very numerical errors due to very
                # large pre-clip actions, we cap pre-clip values
                # hard at 10 times the bounds.
                # This can happen if an action has no effect
                # on the gains process (e.g. hedge == 0)
                actions = tf.minimum( actions, ubnd_a*self.outer_clip_cut_off, name="outer_clip_min" )
                actions = tf.maximum( actions, lbnd_a*self.outer_clip_cut_off, name="outer_clip_max" )

            dbnd = ubnd_a - lbnd_a
            actions  = tf.debugging.check_numerics(actions, "Numerical actions error before clipping action in %s. Turn on tf.enable_check_numerics to find the root cause." % __file__ )
            rel  = ( actions - lbnd_a ) / dbnd
            act  = tf.debugging.check_numerics(rel, "Numerical error before clipping action in %s. Turn on tf.enable_check_numerics to find the root cause." % __file__ )
            rel  = self.softclip( rel )
            act  = tf.where( dbnd > 0., rel *  dbnd + lbnd_a, 0., name="soft_clipped_act" )
            act  = tf.debugging.check_numerics(act, "Numerical error clipping action in %s. Turn on tf.enable_check_numerics to find the root cause." % __file__ )
            return act
