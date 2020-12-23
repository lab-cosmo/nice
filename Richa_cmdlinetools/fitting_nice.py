import pickle
import sys, os, argparse
import ase
import ase.io as ase_io
import numpy as np
import tqdm
import json
import math
from nice.blocks import *
from nice.utilities import *

def main():
    """
    Command-line utility to optimize features. The output is a numpy array of hyper parameters and NICE model. We need the following from the user:
    1. The database file. 
    2. The name for final output file.
    3. Index for training commands for ase.io.read commands.
    4. The number of environments to fit nice transfomers
    5. Input for HYPERS parameters. Keeping 'gaussian_sigma_type': 'Constant','cutoff_smooth_width': 0.3, and 'radial_basis': 'GTO'
    6. Input for standardblocks: num_expand, max_take, and n_components for covariants and invariants. Can be a single number (to be considered for all blocks) or a list (separate for each block)
     7. The desired number of blocks in the StandardBlocks. The last block only considers invariants irrespective of the user entry.
     8. Any additional hypers (json)
                    
    """
        
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=22)
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=formatter)
    parser.add_argument('input', type=str, default="", nargs="?", help='XYZ file to load')
    parser.add_argument('-o', '--output', type=str, default="", help='Output files prefix. Defaults to input filename with stripped extension')
    parser.add_argument('-w','--which_output', type=int, default=1, help='1 for getting a different NICE for each species or else a single NICE for all species')
    parser.add_argument('--train_subset', type=str, default="0:10000", help='Index for reading the file for training in ASE format')
    parser.add_argument('--environments_for_fitting', type=int, default=1000, help='Number of environments for fitting')
    parser.add_argument('--interaction_cutoff', type=float, default=6.3, help='Interaction cut-off')
    parser.add_argument('--max_radial', type=int, default=5, help='Number of radial channels')
    parser.add_argument('--max_angular', type=int, default=5, help='Number of angular momentum channels')
    parser.add_argument('--gaussian_sigma_constant', type=float, default=6.3, help='Gaussian smearing')
    parser.add_argument('--numexpcov', type=str, default=None, required=True, help='Number of the most important input pairs for expansion of covariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--numexpinv', type=str, default=None, required=True, help='Number of the most important input pairs for expansion of invariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--maxtakecov', type=str, default=None, required=True, help='Number of features to be considered for purification step for covariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--maxtakeinv', type=str, default=None, required=True, help='Number of features to be considered for purification step for invariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--ncompcov', type=str, default=None, required=True, help='Number of components for the PCA step for covariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--ncompinv', type=str, default=None, required=True, help='Number of components for the PCA step for invariants. Either a list "Num,Num,Num" or a number for each block "Num".')
    parser.add_argument('--blocks', type=int, default=1, help='Number of blocks that run the NICE sequence')
    parser.add_argument('--json', type=str, default='{}', help='Additional hypers, as JSON string')
        
    args = parser.parse_args()
    
    #File inputs
    filename = args.input
    output   = args.output
    whichoutput = args.which_output
    
    #ASE read inputs
    train_subset = args.train_subset
    environments_for_fitting = args.environments_for_fitting
    
    #Hyper inputs
    ic = args.interaction_cutoff
    n = args.max_radial
    l = args.max_angular
    sig = args.gaussian_sigma_constant
    json_hypers = json.loads(args.json)
    nblocks = args.blocks
    
    #get_NICE inputs
    list_numexpcov  = [item for item in args.numexpcov.split(',')]
    list_numexpinv  = [item for item in args.numexpinv.split(',')]
    list_maxtakecov = [item for item in args.maxtakecov.split(',')]
    list_maxtakeinv = [item for item in args.maxtakeinv.split(',')]
    list_ncompcov   = [item for item in args.ncompcov.split(',')]
    list_ncompinv   = [item for item in args.ncompinv.split(',')]
    
    numexpcov  = []
    numexpinv  = []
    maxtakecov = []
    maxtakeinv = []
    ncompcov   = []
    ncompinv   = []
    
    if len(list_numexpcov ) == 1:
        for nu in range(0, nblocks-1):
            numexpcov.append(int(list_numexpcov[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_numexpcov[nu] == "" or list_numexpcov[nu] == "None":
                numexpcov.append(None)
            else:
                numexpcov.append(int(list_numexpcov[nu]))
    
    if len(list_numexpinv ) == 1:
        for nu in range(0, nblocks-1):
            numexpinv.append(int(list_numexpinv[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_numexpinv[nu] == "" or list_numexpinv[nu] == "None":
                numexpinv.append(None)
            else:
                numexpinv.append(int(list_numexpinv[nu]))
    
    if len(list_maxtakecov)==1:
        for nu in range(0,nblocks-1):
            maxtakecov.append(int(list_maxtakecov[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_maxtakecov[nu] == "" or list_maxtakecov[nu] == "None":
                maxtakecov.append(None)
            else:
                maxtakecov.append(int(list_maxtakecov[nu]))
    
    if len(list_maxtakeinv)==1:
        for nu in range(0,nblocks-1):
            maxtakeinv.append(int(list_maxtakeinv[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_maxtakeinv[nu] == "" or list_maxtakeinv[nu] == "None":
                maxtakeinv.append(None)
            else:
                maxtakeinv.append(int(list_maxtakeinv[nu]))
                
    if len(list_ncompcov)==1:
        for nu in range(0,nblocks-1):
            ncompcov.append(int(list_ncompcov[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_ncompcov[nu] == "" or list_ncompcov[nu] == "None":
                ncompcov.append(None)
            else:
                ncompcov.append(int(list_ncompcov[nu]))
    
    if len(list_ncompinv)==1:
        for nu in range(0,nblocks-1):
            ncompinv.append(int(list_ncompinv[0]))
    else:
        for nu in range(0, nblocks-1): 
            if list_ncompinv[nu] == "" or list_ncompinv[nu] == "None":
                ncompinv.append(None)
            else:
                ncompinv.append(int(list_ncompinv[nu]))
    
    #Output file
    if output == "":
        output = os.path.splitext(filename)[0]
        print(output)
        
        
    #Some constants    
    HARTREE_TO_EV = 27.211386245988
    
    #Building HYPERS
    HYPERS = { **{
    'interaction_cutoff': ic,
    'max_radial': n,
    'max_angular': l,
    'gaussian_sigma_type': 'Constant',
    'gaussian_sigma_constant': sig,
    'cutoff_smooth_width': 0.3,
    'radial_basis': 'GTO'
    }, **json_hypers }
  
    """
    Now that we have all the inputs in place, the next step is to build a StandardSequence.
    StandardSequence --> Block implementing logic of the main NICE sequence.
    StandardBlock --> Block for standard procedure of body order increasement step for covariants and invariants.
    a. ThresholdExpansioner --> Covariant [Block to do Clebsch-Gordan iteration. It uses two even-odd pairs of Data instances with covariants to produce new ones. If first even-odd pair contains covariants of body order v1, and the second v2, body
    order of the result would be v1 + v2.]
    b. CovariantsPurifierBoth --> Covariant [Block to purify covariants of both parities. It operates with pairs of instances of Data class with covariants]
    c. IndividualLambdaPCAsBoth --> Covariant [Block to do pca step for covariants of both parities. It operates with even-odd pairs of instances of Data class]
    d. ThresholdExpansioner --> Invariant [Block to do Clebsch-Gordan iteration.]
    e. InvariantsPurifier --> Invariant [Block to purify invariants. It operates with numpy 2d arrays containing invariants]
    f. InvariantsPCA --> Invariant [Block to do pca step for invariants. It operates with 2d numpy arrays]

    """
    def get_nice():
        numax = nblocks;
        sb = [ ]
        for nu in range(1, numax-1): # this starts from nu=2 actually
            sb.append(
                StandardBlock(ThresholdExpansioner(num_expand=numexpcov[nu-1]),
                      CovariantsPurifierBoth(max_take=maxtakecov[nu-1]),
                      IndividualLambdaPCAsBoth(n_components=ncompcov[nu-1]),
                      ThresholdExpansioner(num_expand=numexpinv[nu-1], mode='invariants'),
                      InvariantsPurifier(max_take=maxtakeinv[nu-1]),
                      InvariantsPCA(n_components=ncompinv[nu-1])) 
                         )
            print("This is expansion ", ThresholdExpansioner(num_expand=numexpcov[nu-1]), "for nu", nu)
        sb.append(
                StandardBlock(None, None, None,
                         ThresholdExpansioner(num_expand=numexpinv[numax-2], mode='invariants'),
                         InvariantsPurifier(max_take=maxtakeinv[numax-2]),
                         InvariantsPCA(n_components=ncompinv[numax-2])) 
                         )
        print("This is a standard block ", sb)
        return StandardSequence(sb,initial_scaler=InitialScaler(mode='signal integral', individually=True))
    
    
    
    #Reading the file for training
    print("Loading structures for training ", filename, " frames: ", train_subset)    
    train_structures = ase.io.read(filename, index=train_subset)
    all_species = get_all_species(train_structures)
    print("These are all the specie indexes in the structure ", all_species)
    
    """
    [environmental index, radial basis/neighbor specie index, lambda, m] with spherical expansion coefficients for 
        environments around atoms with specie indicated in key.
    """
    
    train_coefficients = get_spherical_expansion(train_structures, HYPERS, all_species)
    print("------------------------------------------------------------------------------- ")
    print("These are the spherical expansion coefficients ", train_coefficients.keys)
    
    if args.which_output:
        #individual nice transformers for each atomic specie in the dataset
        nice = {}
        for key in train_coefficients.keys():
            nice[key] = get_nice()
        for key in train_coefficients.keys():
            nice[key].fit(train_coefficients[key][:environments_for_fitting])
        
        print("This is the model ", nice)
       
    else:
        #a single nice transformer irrespective of the specie
        #1. Take all coefficients for each specie and merge all the coefficients together.Then based on the user-defined number of environments to be used for fitting, we choose the required number of coefficients.
        nice = {}
        all_coefficients = [train_coefficients[key] for key in train_coefficients.keys()]
        all_coefficients = np.concatenate(all_coefficients, axis=0)
        np.random.shuffle(all_coefficients)
        all_coefficients = all_coefficients[0:environments_for_fitting]
        #2. Use the model to fit nice on the coefficients chosen above
        nice_single = get_nice()
        nice_single.fit(all_coefficients)
        #3. Irrespective of the central specie, we use the same nice transformer
        nice = {specie: nice_single for specie in all_species}
    
    print("Dumping NICE model as a numpy array")
    with open(output+".npy", 'wb') as f:
        np.save(f, HYPERS)
        np.save(f, nice)
    
if __name__ == '__main__':
    main()      