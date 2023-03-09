# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:24:52 2020
@author: hansb
"""

import unittest
import pickle
import cdx_tf.util as util
from cdxbasics.logger import Logger

import numpy as np
import pandas as pd

util._log.setLevel(Logger.CRITICAL+1)   # no logging in testing

class cdx_tf_basicxs(unittest.TestCase):

    def test_cdx_tf(self):

        pass
        
if __name__ == '__main__':
    unittest.main()


