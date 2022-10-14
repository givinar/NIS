import os
import sys
import math
import numpy as np
import torch
import argparse

import functions
# os.system("export LD_LIBRARY_PATH=$HOME/MultiNest/lib:$HOME/cuba/:$LD_LIBRARY_PATH")

from pycuba import *

def parserCuba():
    parser = argparse.ArgumentParser(description='Integration with Cuba')
    parser.add_argument('-f','--function', action='store', required=True, type=str, 
                        help='Name of the function to integrate in functions.py')
    parser.add_argument('-n','--ndim', action='store', required=True, type=int, 
                        help='Number of dimensions')
    parser.add_argument('--nstart', action='store', required=False, type=int, default=1000,
                        help='Number of starting points (default : 1000)')
    parser.add_argument('--nincrease', action='store', required=False, type=int, default=0,
                        help='Increment in number of points at each iteration (default : 0)')
    parser.add_argument('--maxeval', action='store', required=False, type=int, default=100000,
                        help='Maximum number of evaluations (default : 100000)')
    parser.add_argument('--epsrel', action='store', required=False, type=float, default=1e-3,
                        help='Minimum relative accuracy (default : 1e-3)')
    args = parser.parse_args()
    return args

args = parserCuba()

func = getattr(functions,args.function)(args.ndim)

def functionForIntegration(ndim, xx, ncomp, ff, userdata):
    x = [xx[i] for i in range(ndim.contents.value)]
    print('Point '+'  '.join([str(i) for i in x]))
    inp = torch.tensor([x])
    ff[0] = func(inp)
    return 0

if __name__ == "__main__":
    result = Vegas(functionForIntegration, 
                   args.ndim, 
                   verbose               = 2,
                   nstart                = args.nstart,
                   maxeval               = args.maxeval,
                   nbatch                = 1000,
                   nincrease             = args.nincrease,
                   epsrel                = args.epsrel)
    
    print ('Results',result['results'][0])
    

