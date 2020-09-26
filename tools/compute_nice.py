#!/usr/bin/python3

import pickle
import sys, os, argparse
import ase.io as ase_io
import numpy as np
import tqdm
from nice.transformers import *
from nice.rascal_coefficients import get_rascal_coefficients_parallelized


def main():
    """
    Command-line utility to compute NICE features given a precomputed pickle.
    """
    
    # Tweak the autogenerated help output to look nicer
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=22)
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=formatter)

    parser.add_argument('input', type=str, default="", nargs="?",
                        help='XYZ file to load')
    parser.add_argument('-o', '--output', type=str, default="",
                        help='Output files prefix. Defaults to input filename with stripped extension')
    parser.add_argument('--select', type=str, default=":",
                        help='Selection of input frames. ASE format.')
    parser.add_argument('--nice', type=str, default="nice.pickle",
                        help='Definition of the NICE contraction. Output from optimize_nice.py')
    parser.add_argument('--blocks', type=int, default=1,
                        help='Number of blocks to break the calculation into.')
    

    args = parser.parse_args()
    
    filename = args.input
    output = args.output
    select = args.select    
    nice = args.nice
    nblocks = args.blocks
    
    if output == "":
        output = os.path.splitext(filename)[0]
        
    print("Loading structures ", filename, " frames: ", select)
    frames = ase_io.read(filename, index=select)
    for f in frames:
        if f.cell.sum() == 0.0:
            f.cell = [100,100,100]
        f.pbc = True
        f.wrap(eps=1e-12)
        
    aa = pickle.load(open(nice, "rb"))
    hypers = aa["HYPERS"]
    nice_sequence = aa["NICE"]
    
    mask = hypers["reference-mask"]
    try:
        mask_id = hypers["reference-mask-id"]
    except:
        mask_id = -1
    lmax = hypers["max_angular"]
    scale = hypers["scale"]
    
    # removes keys that are not needed for rascal
    for k in ["nsph", "keep", "keep0", "screen", "screen0", "pcond", "scale",
              "reference-file", "reference-sel", "reference-mask", "reference-mask-id"]:
        hypers.pop(k)
    print("HYPERPARAMETERS ", hypers)
            
    print("Precomputing CG coefficients")
    cglist = ClebschGordan(lmax)
    
    lblock = len(frames)//nblocks
    featinv = {}
    for iblock in tqdm.tqdm(range(nblocks)):
        print("Computing spherical expansion")
        bstart = iblock*lblock
        bend = (iblock+1)*lblock
        if iblock == nblocks-1:
            bend = len(frames)
        coefficients = get_rascal_coefficients_parallelized(frames[bstart:bend], hypers, mask=(None if mask=="" else mask), mask_id=(None if mask_id<0 else mask_id))
        # merge all the coefficients, as we want (for some reason) to apply the same NICE to all species. if mask has been specified, this does nothing
        l = []
        for f in coefficients.values():
            l.append(f)
        coefficients = np.vstack(l)    
        coefficients *= scale

        invariants_even = nice_sequence.transform(coefficients, return_only_invariants = True)
        
        for k in invariants_even:
            if k in featinv:
                featinv[k].append(invariants_even[k]) 
            else:
                featinv[k] = [invariants_even[k]]
        
    print("Saving to ", output)
    for k in featinv:
        featinv[k] = np.vstack(featinv[k])
    pickle.dump(featinv, open(output, "wb"))
        
if __name__ == '__main__':
    main()
            
    
                        
           
