# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:24:52 2020
@author: hansb
"""

import unittest
import pickle
import cdxtfbasics.util as util

import numpy as np
import pandas as pd

mdl_subdir._log.setLevel(Logger.CRITICAL+1)   # no logging in testing

class cdx_tf_basicxs(unittest.TestCase):

    def test_dctkwargs(self):
 
        def f1(**kwargs):
            kwargs = dctkwargs(kwargs)
            a = kwargs('a',1)      # with default
            b = kwargs['b']        # no default; must exist
            c = kwargs.get('c',3)  # with default
            kwargs.done()
            return (a,b,c)
    
        self.assertEqual((-1,-2,-3), f1(a=-1,b=-2,c=-3))
        self.assertEqual((+1,-2,+3), f1(b=-2))
        with self.assertRaises(KeyError):
            f1() # missing b
        
        
if __name__ == '__main__':
    unittest.main()


