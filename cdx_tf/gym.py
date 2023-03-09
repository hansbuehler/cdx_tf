"""
Keras Model base class for Gym models
-------------------------------------
* Cachable
* Keeps best state
* Optimized for async training

Feb 25, 2023
@author: hansbuehler
"""
from collections.abc import Mapping
from cdxbasics.config import Config, Int, Float
from cdxbasics.verbose import Context
from cdxbasics.logger import Logger
from cdxbasics.prettydict import PrettyDict as PrettyDict
from cdxbasics.util import uniqueHash, fmt_now, fmt_seconds
from cdxbasics.subdir import SubDir, uniqueFileName48, CacheMode
from .util import npCast, tfCast, def_dtype
from .optimizer import create_optimizer
import tensorflow as tf
import numpy as np
import time as time

_log = Logger(__file__)

dtype = def_dtype

MODEL_FILE = "gym.h5"
PROGRESS_FILE = "progress"

# ==========================================================================
# Model
# Cachable gym
# ==========================================================================

class Gym(tf.keras.Model):
    """ 
    Base class for a keras gym model with
    * Automatic caching
    * Default tracking of progress, with support for asynchronous training

    This gym assumes that the loss of the gym is "linear", e.g. an expectation of a returned variable (usually 'loss')
    
    Implementation comments
    -----------------------
    In order to implement your own gym,
        1) derive from gym
        2) implement build/call.
           The data passed to each will be of the form of the dictionaries in environment.trn.tf_data.
        3) Set the class variable CACHE_VERSION to a string version
        4) Provide a static LOSS_NAME member if the primary loss objective is not called 'loss' (e.g. the data in your returned dictionary)
        4) Implement from_config / get_config. Without doing this, the system cannot create/restore the gym from a cache
    """
    
    CACHE_VERSION = "0.0.1"
    LOSS_NAME     = "loss" 
    
    def __init__(self, config          : Config,
                       name            : str = None,
                       dtype           : tf.DType = None,
                       trainable       : bool = True,
                       cache_uid       : str = None ):
        """
        Initializes the cachable gym
        ------------------------------
            config : Config
                Configuration for the model.
                Unless cache_uid is provided, this determines the unique ID of the gym.
            name : str
                Name of the object
            dtype : tf.DType
                Type of the gym
            trainable : bool
                Whether the gym is trainable.
            cache_uid : int
                Unique ID for this model setup. If not provided, config.unique_id will be used.
        """
        tf.keras.Model.__init__(self, name=name, dtype=dtype, trainable=trainable )

        _log.verify( isinstance(config, Config), "'config' must be of type '%s' but appears to be of type '%s'", Config.__name__, type(config).__name__ )
        self._cache_unique_id = config.unique_id() if cache_uid is None else str(cache_uid)
        self._cache_config    = config.copy() 
        
    # -------------------
    # syntatic sugar
    # -------------------
    
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The gym must have been call()ed once """
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] ) if not weights is None else 0.
        
    @property
    def cache_uid( self ):
        """ Return the unique ID for this gym. """
        return self._cache_unique_id
    
    @property
    def cache_def_directory_name( self ):
        """ Returns a descriptive name for this class which can be used as directory for the caches. No trailing '/' """
        name = str( self.__class__.__name__ )
        return name if self._cache_version is None else ( name + "/" + self._cache_version )

    # -------------------
    # Keras serialization    
    # -------------------

    @staticmethod
    def from_config( tf_config : dict ):
        """
        You will have to implement this for all derived classes.
        Default pattern is written below        
        """
        raise NotImplementedError("Please implement Gym.from_config")
        
    @staticmethod
    def from_config_default( Class : type, tf_config : dict ):
        """
        Default 'from_config' code.
        Meant to be used as
        
        class MyGyum(Gym):
            
            @staticmethod
            def from_config( tf_config : dict ):
                return Gym.from_config_default( MyGym, tf_config )
        """
        _log.verify( tf_config['cache_version'] == CACHE_VERSION, "Error reading configuration from cache for '%s'': version mismatch. Found version %s, but current code is version %s", Class.__name__, tf_config['cache_version'], Class.CACHE_VERSION )        
        return Class( config        = tf_config['config'],
                      name          = tf_config['name'],
                      dtype         = tf_config['dtype'],
                      trainable     = tf_config['trainable']
                )
        
    def get_config( self ):
        """ You will need to overwrite this if your __init__ signature changes """
        return dict(
            config        = self._cache_config,
            name          = self.name,
            dtype         = self.type,
            trainable     = self.trainable,
            cache_version = self.CACHE_VERSION
            )


# ==========================================================================
# TrainingInfo
# Information on the current training run
# ==========================================================================

class TrainingInfo(object):
    """
    Information on the current training run for user updates
    """
    
    def __init__(self, *, batch_size, epochs ):#NOQA
        self.epochs       = epochs       # total epochs requested by the user. Note that the actual epochs will be reduced by the number of epochs already trained in the cached file
        self.batch_size   = batch_size   # batch size.

# ==========================================================================
# Environment
# Contains the top level data available throughout the training process
# ==========================================================================

class Environment( PrettyDict ):
    """ 
    Represents the data available for the overall training loop: the gym model, its data, sample weights.
    This means this environment can also execute a predict() on the current gym for both its training and validation set.
    This is implemented in predict().
    
    Objects of this class are not serialized directly.
    
    The usual step is to create one's own, e.g. to add additional environment data
    """
    
    def __init__(self, *, gym                : Gym,
                          tf_trn_data        : dict,
                          tf_val_data        : dict = None,
                          trn_sample_weights : np.ndarray = None,
                          val_sample_weights : np.ndarray = None,
                          **kwargs ):
        """
        Initialize environment.
        The gym is set as part of training via set_model (e.g. if the gym was restored from a cache)
        
        Parameters
        ----------
            gym : Gym
                Instance of a gym derived from Gym.
                Note that in case the gym is restored from the cache this object is replaced via set_gym().
            tf_trn_data : dict
                Dictionary of TF data to be passed to the gym during training.
                If the sample path are distributed according to some sample_weights,
                then this dictionary must contain the probabiltiy weights and key_sample_weights must
                be set to the name of this element.
            tf_val_data : dict
                Dictionary of TF data used for validation. Set to None for no validation
            trn_sample_weights : np.ndarray
                Sample weights for the training data set. None for the uniform distribution.
            val_sample_weights : np.ndarray
                Sample weights for the validation data set. None for the uniform distribution.
            **kwargs : 
                Other arguments to be passed to 'self', see PrettyDict.
                In particular, this allows assigning member values to the environment as follows:
                
                    e = Environment(gym, trn_data, user_data = user_data )
                    
                in this case  'user_data' is available in the environment in all subsequent contexts                
        """
        _log.verify( not gym is None, "'gym' cannot be None")
        _log.verify( isinstance(gym, Gym), "'gym' must be derived from 'Gym'. Found type '%s'", type(gym).__name__)
        _log.verify( isinstance( tf_trn_data, Mapping), "tf_trn_data must be a Mapping. Found type '%s'", type(tf_trn_data).__name__ if not tf_trn_data is None else "None")
        if not tf_val_data is None: _log.verify( isinstance( tf_val_data, Mapping), "tf_val_data must be a Mapping. Found type '%s'", type(tf_val_data).__name__)
        
        self.gym                = gym
        self.loss_name          = gym.LOSS_NAME
        self.trn                = PrettyDict()
        self.trn.tf_data        = tfCast( tf_trn_data )
        self.trn.sample_weights = np.asarray( trn_sample_weights ) if not trn_sample_weights is None else None
        if not self.trn.sample_weights is None:
            self.trn.sample_weights = self.trn.sample_weights[:,0] if len(self.trn.sample_weights) == 2 and self.trn.sample_weights.shape[1] == 1 else self.trn.sample_weights
            _log.verify( len(self.trn.sample_weights.shape) == 1, "'trn_sample_weights' must be a vector or of shape (N,) or (N,1), but found tensor of shape %s", trn_sample_weights.shape)

        if tf_val_data is None:
            self.val = None
        else:
            self.val                = PrettyDict()
            self.val.tf_data        = tfCast( tf_val_data )
            self.val.sample_weights = np.asarray( val_sample_weights ) if not val_sample_weights is None else None
            _log.verify( self.trn.sample_weights is None == self.val.sample_weights is None, "'val_sample_weights' and 'trn_sample_weights' must be specified jointly, or jointly omitted")

            if not self.val.sample_weights is None:
                self.val.sample_weights = self.val.sample_weights[:,0] if len(self.val.sample_weights) == 2 and self.val.sample_weights.shape[1] == 1 else self.val.sample_weights
                _log.verify( len(self.trn.sample_weights.shape) == 1, "'val_sample_weights' must be a vector or of shape (N,) or (N,1), but found tensor of shape %s", val_sample_weights.shape)

        if len(kwargs) > 0:
            self.update(kwargs)
            
    def set_gyml(self, gym : Gym, cached : bool ):
        """
        Called to set the gym of the environment.
        This function can be overwritten to handle additional process with the gym
        
        Parameters
        ----------
            gym : Model
                The gym
            cached : bool
                Indicates whether the gym has been restored from disk
        """
        _log.verify( isinstance( gym, Gym ), "Cannot call set_model(): gym must be derived from Gym. Found type '%s'", type(gym).__name__ if not gym is None else "None" )
        self.gym = gym
        
    def predict(self):
        """
        Call current gym on tf_data and tf_val_data to predict the latest results of the gym
        
        Returns
        -------
            A PrettyDict which contains
                trn.result : numpy arrays of the training results from gym(trn.tf_data)
                trn.loss   : float of the training loss for the current gym               
            If val is not None:
                val.result : numpy arrays of the validation results from gym(val.tf_data)
                val.loss   : float of the validation loss for the current gym               
        """
        # training set
        pack              = PrettyDict()
        pack.trn          = PrettyDict()        
        pack.trn.results  = npCast( self.gym(self.trn.tf_data) )
        _log.verify( isinstance(pack.trn.results, np.ndarray) or ( isinstance(pack.trn.results, Mapping) and self.loss_name in pack.trn.results), "The data returned from the gym must either be the loss tensor, or be a dictionary with '%s' entry as specified by '%s.LOSS_NAME'. Model returned data type %s", self.loss_name, type(self.gym).__name__, str(type(pack.trn.results)))

        pack.trn.loss     = pack.trn.results if isinstance(pack.trn.results, np.ndarray) else pack.trn.results[self.loss_name]
        pack.trn.loss     = pack.trn.loss[:,0] if len(pack.trn.loss.shape) == 2 and pack.trn.loss.shape[1] == 1 else pack.trn.loss
        _log.verify( len(pack.trn.loss.shape) == 1, "'loss' must be a vector or of shape (N,1). Found tensor of shape %s", pack.trn.loss.shape)
        if not self.trn.sample_weights is None:
            _log.verify( len(pack.trn.loss) == len(self.trn.sample_weights), "Invalid training sample weight vector: loss vector returned by gym is of length %ld, while training sample weights are of length %ld", len(pack.trn.loss), len(self.trn.sample_weights))        
        pack.trn.loss    = np.sum( self.trn.sample_weights * pack.trn.loss ) if not self.trn.sample_weights is None else np.mean( pack.trn.loss )     

        # validation set        
        if self.val is None:
            pack.val = None
        else:
            pack.val          = PrettyDict()
            pack.val.results  = npCast( gym(tf_val_data) ) 
            pack.val.loss     = pack.val.results if isinstance(pack.val.results, np.ndarray) else pack.val.results[self.loss_name]
            pack.val.loss     = pack.val.loss[:,0] if len(pack.val.loss.shape) == 2 and pack.val.loss.shape[1] == 1 else pack.val.loss
            pack.val.loss     = np.sum( self.val.sample_weights * pack.val.loss ) if not self.val.sample_weights is None else np.mean( pack.val.loss )     

        return pack
        
# ==========================================================================
# ProgressData
# Base class for relevant data to be computed during training for user
# feedback (e.g. history of losses; current state of the gym)
# ==========================================================================

class ProgressData(object):
    """
    Base class for relevant data to be computed during training for user
    feedback (e.g. history of losses; current state of the gym).
    
    This class is intended to be derived from, and that you overwrite on_epoch_end.
    
    For being used in Ray, this class needs to be pickle'able.
    """
    
    STOP_CONVERGED   = -1
    STOP_ABORTED     = -2
    CONTINUE         = 0
    
    def __init__(self, environment     : Environment,        # gym, tf_data, etc
                       training_info   : TrainingInfo,       # total number of epochs requested etc
                       predicted_data0 : PrettyDict         
                       ):
        """
        Initialize the cachable progress data store
        ** Do not store the gym or the training data into this object **
        
        Parameters
        ----------
            environment : Environment,
                provides access to various non-serializable objects in the training loop
            epochs : int
                Number of epochs to be computed.
            predicted_data0 : PrettyDict
                Result of environment.predict(). If None, this will be computed on-the-fly.
                
        """
        self.times_per_epoch = []
        self.trn_losses      = [ predicted_data0.trn.loss ]
        self.val_losses      = [ predicted_data0.val.loss ] if not predicted_data0.val is None else None
        
        # best epoch
        self.best_epoch      = -1
        self.best_weights    = environment.gym.get_weights()
        self.best_loss       = predicted_data0.trn.loss
    
    @property
    def current_epoch(self):
        """ Returns the current epoch. Returns -1 if no epoch was yet recorded """
        return len(self.times_per_epoch)-1
    
    def on_epoch_end(self,  environment    : Environment,  # gym, tf_data, etc
                            predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                            logs           : dict          # logs c.f. keras Callback
                        ) -> int:
        """ 
        Callback at the end of an epoch. Typically used to update visuals.
        Return self.STOP_CONVERGED or self.STOP_ABORTED to abort training or self.CONTINUE to continue
        """
        return self.CONTINUE
    
    def on_done(self,       environment    : Environment,  # gym, tf_data, etc
                            predicted_data : PrettyDict,   # current predicted training and validation data; current loss.
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                        ):
        """
        Called when training is finished and the gym was set to the best weights
        Typically used to update any visualization and print a summary.
        """
        pass
        
    # --------------------
    # Internal
    # --------------------
            
    def _on_epoch_end(self, environment    : Environment,  # gym, tf_data, etc
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                            time_epoch     : float,        # time required for 
                            logs           : dict          # logs c.f. keras Callback
                        ):
        """
        Called at the end of an epoch by Callback()
        Will store the time for the epoch in 'times_per_epoch'
        
        This function is called by the training loop.
        Do not overwrite this function; instead overwrite on_epoch_end()
        """
        assert len(self.times_per_epoch)+1 == len(self.trn_losses), "Internal error: %ld+1 != %ld" % (len(self.times_per_epoch), len(self.trn_losses))

        predicted_data = environment.predict()

        self.times_per_epoch.append( time_epoch )
        self.trn_losses.append( predicted_data.trn.loss )
        if not self.val_losses is None:
            self.val_losses.append( predicted_data.val.loss )
        
        if self.best_loss > predicted_data.trn.loss:
            self.best_epoch   = self.current_epoch
            self.best_weights = environment.gym.get_weights()
            self.besr_loss    = predicted_data.trn.loss
        
        return self.on_epoch_end( environment=environment, predicted_data=predicted_data, training_info=training_info, logs=logs )
        
    def _on_done(self,      environment    : Environment,  # gym, tf_data, etc
                            training_info  : TrainingInfo, # number of epochs to be computed etc
                        ):
        """ 
        Called by the Callback() when training is done, after the best weights have been written to the gym.
        Do not overwrite this function; overwrite on_done()
        """
        predicted_data = environment.predict()
        self.on_done( environment=environment, predicted_data=predicted_data, training_info=training_info )
        
    def _write_best( self,  gym : Gym ):
        """ Write best weights to the gym, if any """
        if self.best_epoch >= 0:
            gym.set_weights( self.best_weights )
        
# ==========================================================================
# Callback
# This is called during training to handle caching and user updates
# ==========================================================================

class Callback(tf.keras.callbacks.Callback):
    """
    Manages training of our gym    
    -- Keeps track of training data in TrainingProgressData including best fit
    -- Implements caching
    -- Implements dyanmic visual updates
    """
    
    STOP_ABORTED     = ProgressData.STOP_ABORTED
    STOP_CONVERGED   = ProgressData.STOP_CONVERGED
    CONTINUE         = ProgressData.CONTINUE
    STOP_INTERRUPTED = -10
    FINISHED_EPOCHS  = 1
    ALREADY_TRAINED  = 2
    
    def __init__(self, *, environment    : Environment,
                          training_info  : TrainingInfo,
                          progress_data  : ProgressData,
                          cache_config   : PrettyDict,
                          verbose        : Context = Context() ):
        """
        Initialize the call back
        The object will attempt to restore a previous training state from disk if found.
        
        Parameters
        ----------
            model_cachable
                Model derived from Model.
            epochs : int
                Total number of epochs for this training. If the cached object has been trained beyond that point, no further training will commence.
            default_cache_directory : str
                Default caching directory for 
        """
        
        tf.keras.callbacks.Callback.__init__(self)
        
        gym                  = environment.gym
        _log.verify( isinstance(gym, Gym), "'gym' must be derived from 'Gym'")

        # basics
        self.environment      = environment
        self.training_info    = training_info
        self.progress_data    = progress_data
        self.cache_last_epoch = progress_data.current_epoch
        self.verbose          = verbose
        self.time_start       = time.time()
        self.stop_reason      = self.CONTINUE
        _log.verify( self.training_info.epochs > 0, "'epochs' must be positive. Found %ld", self.training_info.epochs )
        assert not self.is_done, "Internal error: nothing to train?"
        
        # caching
        self.cache_mode       = cache_config.cache_mode
        self.cache_freq       = cache_config.cache_freq
        self.cache_dir        = cache_config.cache_dir
        self.time0            = time.time()

    def write_cache(self):
        """ Write cache to disk """
        if self.cache_last_epoch >= self.progress_data.current_epoch:
            assert self.cache_last_epoch == self.progress_data.current_epoch, "Interal error: %ld > %ld ?" % (self.cache_last_epoch, self.progress_data.current_epoch)
            return
        assert self.progress_data.current_epoch >= 0, "Internal error: current epoch is -1"
        self.cache_dir.write( PROGRESS_FILE, self.progess_data )
        self.environment.gym.save( self.cache_dir + MODEL_FILE )
        self.cache_last_epoch = self.progress_data.current_epoch
        
    @property
    def is_done(self):
        """ Checks whether training has finished. This can happen at inception if a cache is restored which was trained for as many epochs as requested """
        return self.progress_data.current_epoch+1 >= self.training_info.epochs
    
    @property
    def current_epoch(self):
        """ Returns the current epoch. -1 if no epoch was run """
        return self.progress_data.current_epoch
    
    @property
    def epochs(self):
        return self.training_info.epochs

    def on_epoch_begin( self, loop_epoch, logs = None ):#NOQA
        pass
    
    def on_epoch_end( self, loop_epoch, logs = None ):
        """
        Called when an epoch ends
        Will call prorgress_data.on_epoch_end()
        
        Note that 'loop_epoch' is the epoch of the current training run.
        If the state was recovered from a cache, it won't be the logical epoch
        """

        time_now = time.time()
        _current = self.progress_data.current_epoch
        r = self.progress_data._on_epoch_end( environment   = self.environment,
                                              training_info = self.training_info,
                                              time_epoch    = time_now - self.time_start,
                                              logs          = logs )
        assert self.progress_data.current_epoch >= 0, "Internal error: progress_data must update its epoch count"
        assert self.progress_data.current_epoch > _current, ("Internal error: progress_data must update its epoch count", self.progress_data.current_epoch, _current)

        # allow calling progress data to abort training
        if r in [ProgressData.STOP_ABORTED, ProgressData.STOP_CONVERGED]:
            self.write_cache()
            self.stop_reason         = r
            self.gym.stop_training = True

        self.time_start = time_now
        
        if self.current_epoch % self.cache_freq == 0 and self.cache_mode.write and self.progress_data.current_epoch > self.cache_last_epoch:
            self.write_cache()
            
    def finalize( self ):
        """
        Close training. Call this even if training was aborted
        -- Cache the current state
        -- Apply best weight
        """
        # cache current state /before/ we reset gym to its best weights
        # this way we can continue to train from where we left it
        cached_msg = ""
        if self.progress_data.current_epoch >= 0 and self.cache_mode.write:
            self.write_cache()
            cached_msg = " State of training until epoch %ld cached into %s\n" % (self.cache_last_epoch+1, self.full_cache_file)
            
        status = ""
        if self.stop_reason == self.STOP_ABORTED:
            status = "Training aborted"
        elif self.stop_reason == self.STOP_CONVERGED:
            status = "Desired convergence achieved"
        elif self.stop_reason == self.STOP_INTERRUPTED:
            status = "User abort"
        elif self.stop_reason == self.FINISHED_EPOCHS:
            status = "Trained %ld epochs" % self.epochs
        elif self.stop_reason == self.ALREADY_TRAINED:
            status = "Model was already trained for at least %ld epochs" % self.epochs
        else:
            _log.throw("Unknown stopping reason %ld", self.stop_reason)            

        # restore best weights & tell user
        self.progress_data._write_best( self.environment.gym )
        self.progress_data._on_done(  environment   = self.environment,
                                      training_info = self.training_info )
        
        self.verbose.write( "Status: %(status)s.\n"\
                           " Weights set to best epoch: %(best_epoch)ld\n"\
                           "%(cached_msg)s Time: %(time)s",\
                           status=status, 
                           best_epoch=self.progress_data.best_epoch+1,
                           cached_msg=cached_msg,
                           time=fmt_now())

# ==========================================================================
# Main training loop
# ==========================================================================

@tf.function
def default_loss( y_true,y_pred ):     
    """ Default loss: ignore y_true """
    return y_pred

def train(   environment    : Environment,
             create_progress: type = ProgressData,
             config         : Config = Config(),
             verbose        : Context = Context() ):
    """
    Main training loop
    
    Parameters
    ----------
        environment : Environment
            Contains the (initial) gym, training and validation data sets. Also contains sample weights.
            You can provide a derived class if you wish to pass on further information to progess_data.on_epoch_end
            Alternatively, you can pass a dictionary with the required elements to construct an Environment object
            Note: if a gym is loaded back from disk, then environment.set_gym() is called.
        progress_data : ProgressData
            Main callback: the function on_epoch_end() is called at the end of each epoch.
            This function is then intended to compute all other summary statistics required for user feedback doing training.
            The object needs to be pickle'abel if it is intended to be used a multiprocessing environment such as Ray
        config : Config
            Standard config
        verbose :
            Controls level of output.

    Returns
    -------
        A PrettyDict which contains, computed at the best weights:
            gym           : trained gym, set to best weights (according to training data)
                            Note that this object may differ from the original environment.gym if it was restored from a cache
            progress_data : progress data, e.g. a version of ProgressData which contains at the very least the time series of losses, and the best weights
            trn.result    : numpy arrays of the training results from gym(trn.tf_data)
            trn.loss      : float of the training loss for the current gym               
        If val is not None:
            val.result    : numpy arrays of the validation results from gym(val.tf_data)
            val.loss      : float of the validation loss for the current gym               
    """
    verbose.write("Initializing training")
    
    # --------------------
    # Prep & Caching
    # --------------------
        
    # how much to print
    debug_numerics   = config.debug("check_numerics", False, bool, "Whether to check TF numerics.")
    tf_verbose       = config.debug("tf_verbose", 0, Int>=0, "Verbosity for TensorFlow fit()")
    
    # training parameters    
    batch_size       = config.train("batch_size",  None, help="Batch size")
    epochs           = config.train("epochs",      100, Int>0, help="Epochs")
    run_eagerly      = config.train("run_eagerly", False, help="Keras gym run_eagerly. Turn to True for debugging. This slows down training. Use None for default.")
    tboard_log_dir   = config.train.tensor_board(   "log_dir", "", str, "Specify tensor board log directory. See https://www.tensorflow.org/guide/profiler")
    tboard_freq      = config.train.tensor_board(   "hist_freq", 1, Int>0, "Specify tensor board log frequency. See https://www.tensorflow.org/guide/profiler") 
    tboard_prf_batch = config.train.tensor_board(   "profile_batch", 0, help="Batch used for profiling. Set to non-zero to activate profiling. See https://www.tensorflow.org/guide/profiler") 

    # create gym  
    # In the current design the newly created gym is replaced by the restored gym if found
    gym                      = environment.gym
    optimizer_uid            = config.train.optimizer.unique_id()
    train_uid                = uniqueFileName48( [ gym.cache_uid(), optimizer_uid, gym.CACHE_VERSION ] )
    
    # caching
    def_directory_name       = gym.cache_def_directory_name + "/batch_size_" + (str(batch_size) if not batch_size is None else "default")
    cache_config             = pdct()
    cache_config.cache_mode  = config.caching("mode", CacheMode.ON, CacheMode.MODES, "Caching strategy: %s" % CacheMode.HELP)
    cache_config.cache_freq  = config.caching("epoch_freq", 10, Int>0, "How often to cache results, in number of epochs")
    cache_dir                = config.caching("directory", "./.cache/" + gym.cache_def_directory_name, str, "Caching directory")
    cache_file_name          = config.caching("file_name", "", str, "Allows overwriting the filename for debugging an explicit cached state")
    config.done()

    cache_file               = train_uid if cache_file_name == "" else cache_file_name
    cache_dir                = SubDir( cache_dir, "!" )
    cache_config.cache_dir   = cache_config.cache_dir.subdir( cache_file )
    cache_config.cache_mode  = CacheMode( cache_config.cache_mode )

    training_info = TrainingInfo( batch_size     = batch_size,
                                  epochs         = epochs)

    # --------------------
    # Restore cache
    # --------------------
    
    progress_data            = cache_config.cache_dir.read( PROGRESS_FILE, None )
    if not progress_data is None:
        """ Handle cached gym """
        assert not progress_data.cache_last_epoch is None
        assert progress_data.cache_last_epoch <= progress_data.current_epoch, "Invalid epoch data: %ld %ld" % (progress_data.cache_last_epoch, progress_data.current_epoch)
        config.progress.mark_done()
        config.train.optimizer.mark_done()
                                                                               
        gym  = keras.models.load_model( cache_config.cache_dir.path + MODEL_FILE  )
        epochs = epochs - (progress_data.current_epoch+1)
        verbose.report(1, "Loaded gym from cache %s. Model was trained for %ld epochs.", cache_config.cache_dir.path, progress_data.current_epoch )
        environment.set_model( gym, cached=True )
        
        if epochs <= 0:
            config.done()
            progress_data._write_best(environment.gym)
            verbose.write("Finished training as cached gym is fully trained")
            result               = environment.predict()
            result.progress_data = progress_data
            result.gym           = gym            
            return result
        
        verbose.report(1, "Preparing training for the remaining %ld epochs", epochs )
    else:
        """ No gym cached yet """
        verbose.report(1, "Model generated. Preparing training for %ld epochs", epochs )
        gym.compile(    optimizer        = create_optimizer(config.train.optimizer),
                        loss             = { environment.loss_name : default_loss },
                        weighted_metrics = { environment.loss_name : default_loss },
                        run_eagerly      = run_eagerly)
        environment.set_model( gym, cached=False )
 
        progress_data = create_progress( environment=environment,
                                         training_info=training_info,
                                         config=config.progress )
        
    config.done()
    
    # --------------------
    # Training
    # --------------------
    
    t0             = time.time()
    pack0          = environment.predict()
    verbose.write("Model has %ld trainable weights." % gym.num_trainable_weights)    
    callback       = Callback(    environment     = environment,
                                  training_info   = training_info,
                                  create_progress = create_progress,
                                  config          = config,
                                  verbose         = verbose.sub(1) )
    config.done()

    # train
    # -----
    
    if debug_numerics:
        tf.debugging.enable_check_numerics()
        verbose.report(1, "Enabled automated checking for numerical errors. This will slow down training. Use config.debug.check_numerics = False to turn this off")
    else:
        tf.debugging.disable_check_numerics()
    
    if not callback.is_done:
        assert epochs > (callback.current_epoch+1), "Internal error. callback.is_done failed"
        # tensorboard
        # See https://docs.aws.amazon.com/sagemaker/latest/dg/studio-tensorboard.html

        tboard = None
        if tboard_log_dir != "":
            t0             = time.time()
            tboard_log_dir = SubDir(tboard_log_dir).path
            tboard         = tf.keras.callbacks.TensorBoard(log_dir=tboard_log_dir, histogram_freq=tboard_freq, profile_batch=tboard_prf_batch )
            verbose.report(1,"TensorBoard log directory set to '%s'. Took %s" % (tboard_log_dir, fmt_seconds(time.time()-t0)))

        def find_sample_size( x ):
            if isinstance(x, tf.Tensor):
                assert int(x.shape[0])>0, x.shape
                return int(x.shape[0])
            if isinstance(x, dict):
                for x in x.values():
                    l = find_sample_size(x)
                    if l>0:
                        return l
            else:
                _log.verify( isinstance(x, list), "Cannot determine data sample size for type %s", str(type(x)))
                for x in x:
                    l = find_sample_size(x)
                    if l>0:
                        return l
            return 0
        
        nSamples = find_sample_size(environment.trn.tf_data)
        _log.verify( nSamples > 0, "Cannot determine data sample size: no tensors found")
        y        = tf.zeros((nSamples,), dtype=gym.dtype)

        try:
            gym.fit(        x              = environment.trn.tf_data,
                            y              = y,
                            batch_size     = batch_size,
                            sample_weight  = environment.trn.sample_weights * float(len(environment.trn.sample_weights)) if not environment.trn.sample_weights is None else None,  # sample_weights are poorly handled in TF
                            epochs         = epochs - (callback.current_epoch+1),
                            callbacks      = callback if tboard is None else [ callback, tboard ],
                            verbose        = tf_verbose )
            callback.stop_reason = Callback.FINISHED_EPOCHS            
        except KeyboardInterrupt:
            callback.stop_reason = Callback.STOP_INTERRUPTED

    callback.finalize()
    verbose.report(0, "Training completed. Total training took %s", fmt_seconds(time.time()-t0))
    result = environment.predict()
    result.progress_data = callback.progress_data
    result.gym = gym
    return result




