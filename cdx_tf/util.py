"""
TF basic utilities
------------------
Import this file in all deep hedging files.
June 30, 2022
@author: hansbuehler
"""

from cdxbasics.logger import Logger
from cdxbasics.util import isAtomic
import numpy as np
import tensorflow as tf
import math as math
import inspect as inspect
import datetime as datetime
_log = Logger(__file__)

# -------------------------------------------------
# Manage tensor flow
# -------------------------------------------------

TF_VERSION = [ int(x) for x in tf.__version__.split(".") ]
TF_VERSION = TF_VERSION[0]*100+TF_VERSION[1]
_log.verify( TF_VERSION >= 210, "Tensor Flow version 2.10 required. Found %s", tf.__version__)

TF_NUM_GPU = len(tf.config.list_physical_devices('GPU'))
TF_NUM_CPU = len(tf.config.list_physical_devices('CPU'))

def_dtype = tf.float32  # this is the default srttpe
tf.keras.backend.set_floatx(def_dtype.name)

"""
DIM_DUMMY
Each world data dictionary must have an element with this name, whcih needs to be of dimension (None,1)
This is used in layers.VariableLayer in order to scale the variable up to the number of samples

The reason the dimension is (None,1) and not (None,) is that older Tensorflow versions auto-upscale data
of dimension (None,) to (None,1) anyway.

It's not a pretty construction but there seems to be no other nice way in TF to find our current sample size.
"""                                

DIM_DUMMY = "_dimension_dummy"  
        
# -------------------------------------------------
# TF <--> NP
# -------------------------------------------------

def tfCast( x, native = True, dtype=def_dtype ):
    """
    Casts an object or a collection of objecyts iteratively into tensors.
    Turns all custom dictionaries (such as PrettyDict) into dictionaries unless 'native' is False.
    
    Parameters
    ----------
        x
            object. Can be list of lists of dicts of numpys etc
                - numpy arrays become tenors
                - tensors will be cast to dtype, if required
                - atomic variables become tensor constants
                - None is None
        native : bool, optional
            True
                - lists-types of x's becomes lists of tensors
                - dicts-types of x's becomes dicts of tensors
            False:
                - lists-types of x's stay list-types
                - dicts-types of x's stay dict-types
                            
        dtype : tf.DType, optional
            Overwrite dtype
            
    Returns
    -------
        tensors.
    """
    if x is None:
        return None
    if isinstance(x, tf.Tensor):
        return x if ( dtype is None or x.dtype == dtype ) else tf.convert_to_tensor( x, dtype=dtype )
    if isinstance(x, np.ndarray):
        return tf.convert_to_tensor( x, dtype=dtype )
    if isAtomic(x):
        return tf.constant( x, dtype=dtype )     
    if isinstance(x, dict):
        d = { _ : tfCast(x[_], dtype=dtype) for _ in x }
        return d if native or (type(x) == 'dict') else x.__class__(d)
    if isinstance(x, list):
        l = [ tfCast(x[_], dtype=dtype) for _ in x ]
        return l if native or (type(l) == 'list') else x.__class__(l)
    
    _log.verify( False, "Cannot convert object of type '%s' to tensor", x.__class__.__name__)
    return None

def npCast( x, dtype=None ):
    """
    Casts an object or a collection of objecyts iteratively into numpy arrays.
    
    Parameters
    ----------
        x
            object. Can be list of lists of dicts of tensors etc
                - tensors become numpy arrays (copies !)
                - numpy arrays will be cast into dtype if necessaryt
                - atomic variables become arrays with shape ()
                - lists of x's becomes lists of npCast(x)'s
                - dicts of x's becomes dicts of npCast(x)'s
                - None returns None
            
        dtype : tf.DType, optional
            Overwrite dtype
            
    Returns
    -------
        numpys.
    """
    if x is None:
        return None
    if isinstance(x, tf.Tensor):
        return np.asarray( x, dtype=dtype )
    if isinstance(x, np.ndarray):
        return np.asarray( x, dtype=dtype )
    if isAtomic(x):
        return np.array(x, dtype=dtype )
    if isinstance(x, dict):
        d  = { _ : npCast(x[_], dtype=dtype) for _ in x }
        return d if type(x) == 'dict' else x.__class__(d)
    if isinstance(x, list):
        l = [ npCast(x[_], dtype=dtype) for _ in x ]
        return l if type(l) == 'list' else x.__class__(l)
    
    return  np.asarray( x, dtype=dtype )

# -------------------------------------------------
# TF flattening
# -------------------------------------------------

@tf.function
def tf_back_flatten( tensor : tf.Tensor, target_dim : int = 2) -> tf.Tensor:
    """
    Flattens a tensor while keeping the first 'dim'-1 axis the same.
    
    x = tf.Tensor( np.array((16,8,4,2)) )
    tf_back_flatten( x, dim = 1)   --> shape [16*8*4*2]
    tf_back_flatten( x, dim = 2)   --> shape [16,8*4*2]
    ...
    
    Use case: assume we are given features for an ML model.
    First dimension is number of Samples. Subsequent dimensions might be present, but are not relevant for network construction.
    This function allows flattening it such that the first dimension remains the number of samples, while the rest is flattened.
    
    Use tf_make_dim to also handles whose existing dimension is less than target_dim
        
    Parameters
    ----------
        tensor : tf.Tensor
            A tensor
        target_dim : int
            max dimension of the flattened tensor.
            
    Returns
    -------
        Flat tensor.
    """
    _log.verify( target_dim > 0 and target_dim <= len(tensor.shape), "'target_dim' most be positive and not exceed dimension of tensor, %ld. Found target_dim %ld. If this intended, use 'tf_make_dim' instead", len(tensor.shape), target_dim)     
    if len(tensor.shape) > target_dim:
        splits = [ tf_back_flatten( tensor[...,_], dim=target_dim ) for _ in range(tensor.shape[-1]) ]
        tensor = tf.concat( splits, axis=-1 )    
    return tensor

@tf.function
def tf_make_dim( tensor : tf.Tensor, target_dim : int = 2) -> tf.Tensor:
    """
    Ensure a tensor as a given dimension by either flattening at the end to
    reduce diemnsions, or adding tf.newaxis to increase them.
    
    x = tf.Tensor( np.array((16,8,4,2)) )
    tf_back_flatten( x, dim = 1)   --> shape [16*8*4*2]
    tf_back_flatten( x, dim = 2)   --> shape [16,8*4*2]

    x = tf.Tensor( np.array((16,)) )
    tf_back_flatten( x, dim = 2)   --> shape [16,1]
    
    Parameters
    ----------
        tensor : tf.Tensor
            A tensor
        target_dim : int
            target dimension of the flattened tensor.
            
    Returns
    -------
        Flat tensor.
    """
    try:
        if len(tensor.shape) > target_dim:
            return tf_back_flatten(tensor,target_dim)
        while len(tensor.shape) < target_dim:
            tensor = tensor[...,tf.newaxis]
        return tensor
    except AttributeError as e:
        _log.throw( "Error converting tensor to dimension %ld: %s\nTensor is %s of type %s", target_dim, e, str(tensor), type(tensor) )

