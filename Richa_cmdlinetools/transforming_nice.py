import pickle
import sys, os, argparse
import ase
import ase.io as ase_io
import numpy as np
import json
import tqdm
import math
from nice.blocks import *
from nice.utilities import *

def main():
    """
    Command-line utility to compute NICE features given a precomputed numpy array of Hyper parameters and NICE. Inputs required from the user:
    1. The database file. 
    2. The name for final output file.
    2. Index for ase.io.read commands.
    3. The name for the file containing the numpy array.
    """
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=22)
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=formatter)
    parser.add_argument('input', type=str, default="", nargs="?",help='XYZ file to load')
    parser.add_argument('-o', '--output', type=str, default="",help='Output files prefix. Defaults to input filename with stripped extension')
    parser.add_argument('--index', type=str, default="0:10000", help='Index for reading the file in ASE format')
    parser.add_argument('--nice', type=str, default="nice.pickle", help='Definition of the NICE contraction. Output from fitting.py')
    #parser.add_argument('--blocks', type=int, default=1,help='Number of blocks to break the calculation into.')
    
    args = parser.parse_args()
    
    filename = args.input
    output = args.output
    select = args.index   
    nice = args.nice
    HARTREE_TO_EV = 27.211386245988
    
    if output == "":
        output = os.path.splitext(filename)[0]
        
    print("Loading structures ", filename, " frames: ", select)
    index_structures = ase_io.read(filename, index=select)
    all_species = get_all_species(index_structures)
    
    with open(nice, 'rb') as f:
        hypers1 = np.load(f, allow_pickle=True)
        nice1 = np.load(f, allow_pickle=True)
    
    hypers=hypers1.tolist()
    nice=nice1.tolist()
    
    
    index_features = transform_sequentially(nice, index_structures, hypers, all_species)
    
    
    ''' getting compositional features suitable for linear regression which contains information
    about the number of atoms with particular species in the structure
    '''
    index_c_features = get_compositional_features(index_structures, all_species)
    
    index_features = np.concatenate([index_features, index_c_features], axis=1)
    
    index_energies = [structure.info['energy'] for structure in index_structures]
    index_energies = np.array(index_energies) * HARTREE_TO_EV
    
    with open(output+".npy", 'wb') as f:
        np.save(f, index_features)
        np.save(f, index_c_features)
        np.save(f, index_energies)
        
if __name__ == '__main__':
    main()
    