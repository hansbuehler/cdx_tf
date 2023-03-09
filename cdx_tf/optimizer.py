"""
TF basic utilities
------------------
Import this file in all deep hedging files.
June 30, 2022
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.config import Config, Float, Int
from .util import tf
import inspect as inspect
_log = Logger(__file__)

def create_optimizer( config : Config ):
    """
    Creates an optimizer from a config object
    The keywords accepted are those documented for https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    
    You can use:
        config.optimizer = "adam"
        config.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    """    
    if isinstance( config, str ):
        return tf.keras.optimizers.get(config)

    # new version. Specify optimizer.name
    config    = config.optimizer
    name      = config("name", "adam", str, "Optimizer name. See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers")

    # auto-detect valid parameters
    optimizer = tf.keras.optimizers.get(name)
    sig_opt   = inspect.signature(optimizer.__init__)
    classname = optimizer.__class__
    kwargs    = {}
    
    # all parameters requested by the optimizer class
    for para in sig_opt.parameters:
        if para in ['self','name','kwargs']:
            continue
        default = sig_opt.parameters[para].default
        if default == inspect.Parameter.empty:
            # mandatory parameter
            kwargs[para] = config(para, help="Parameter %s for %s" % (para,classname))
        else:
            # optional parameter
            kwargs[para] = config(para, default, help="Parameter %s for %s" % (para,classname))

    # The following parameters are understood by general tensorflow optimziers
    hard_coded = dict(  clipnorm=None,
                        clipvalue=None,
                        global_clipnorm=None )
    if TF_VERSION >= 211:
        hard_coded.update(
                        use_ema=False,
                        ema_momentum=0.99,
                        ema_overwrite_frequency=None)
    for k in hard_coded:
        if k in kwargs:
            continue  # handled already
        v = hard_coded[k]
        kwargs[k] = config(k, v, help="Parameter %s for keras optimizers" % k)
    
    config.done()
    return optimizer.__class__(**kwargs)


