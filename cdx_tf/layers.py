"""
Various methods for clipping
-----------------------------
March 1st, 2023
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config, Float, Int
from .util import tf, def_dtype
_log = Logger(__file__)

def dense_model( nInput    : int, 
                 nOutput   : int, 
                 config    : Config = Config(),
                 dtype     : tf.DType = def_dtype,
                 name      : str = None,
                 trainable : bool = True,
                 def_width : int = 50,
                 def_depth : int = 3,
                 def_activation : str = "relu",
                 def_final_act  : str = "linear"
                 ) -> tf.keras.Model:                 
    """
    Create simple dense neural network.
    The function creates a dense network of 'depths' and 'width' with activation function 'activation'.
    The final layer then condenses the last 'width' nodes to 'nOutput' nodes, usually with a linear 'final_act'.
    
    Parameters
    ----------
        nInput, nOutput: number of input and output nodes
        config         : user settings:
                                width        : width of the network
                                depth        : depth
                                activation   : activation function in core network
                                final_activation : activation function to compress last 'width' to nOutput. Usually linear.
                                zero_model   : whether the model is initialzed to zero initial value (but not zero gradients)
                                regression   : hard set the network to a pure regression model.
                                               This is equivalent to setting depth = 0, width = nInput, and final_activation = 'linear'.
        dtype,
        name, 
        trainable       : see keras layers
        def_*           : default values.
        
    Returns
    -------
        Keras model.
    """
    
    width      = config("width", def_width, Int>0, "Width")
    depth      = config("depth", def_depth, Int>=0, "Depth")
    activation = config("activation", def_activation, str, "Activation function")
    final_act  = config("final_activation", "linear", str, "Final activation function")
    zero_model = config("zero_model", True, bool, "Whether to initialze levels (but not derivatives) of the model with zero")
    regression = config("simple_regression", False, bool, "Learn simple regression model only")
    config.done()

    if regression:
        depth      = 0
        width      = nInput
        final_act  = 'linear'
    
    def dense(width,activation, x1, x2 ):
        d1 = tf.keras.layers.Dense(units=width,
                                   activation=activation,
                                   use_bias=True, dtype=dtype,
                                   trainable=True )
        if x2 is None:
            return d1(x1), None

        d2 = tf.keras.layers.Dense(units=width,
                                   activation=activation,
                                   use_bias=True, dtype=dtype,
                                   trainable=False )
        x1 = d1(x1)
        x2 = d2(x2)
        return x1, x2
    
    inp       = tf.keras.layers.Input( shape=(nInput,), dtype=dtype )
    x1, x2    = ( inp, inp if zero_model else None )
    for d in range(depth):
        x1, x2    = dense( width, activation, x1, x2 )
    x1, x2 = dense( nOutput, final_act, x1, x2 ) 
    x  = tf.keras.layers.Subtract()([x1, x2]) if not x2 is None else x1
    m  = tf.keras.Model( inputs=inp, outputs=x, name=name )
    if zero_model:
        for v1, v2 in zip( m.trainable_weights, m.non_trainable_weights ):
            v2.assign( v1.value() )        
    return m
  