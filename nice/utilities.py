import tqdm
import numpy as np
import nice.rascal_coefficients
import copy
import os
from multiprocessing import Pool, cpu_count
import warnings


def get_all_species(structures):
    ''' getting all unique atomic species among the structures
    
    Args:
        structures: list of ase atoms objects
        
    Returns:
        sorted numpy array with ints with all unique species in the format where 
        1 states for H, 2 for He and so on. (inherits from ase function 
        atoms_object.get_atomic_numbers())
        
    '''
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)
    all_species = np.sort(np.unique(all_species))
    return all_species


def get_compositional_features(structures, all_species):
    ''' getting compositional features suitable for linear regression which contains information
    about the number of atoms with particular species in the structure
    
    Args:
        structures: list of Ase atoms objects
        all_species: numpy array with ints of all unique species in the dataset. \
        If all species argument is the same for several calls of this function, resulting \
        blocks of compositional features are guaranteed to be consisted with each other
        
    Return:
        numpy array with shape [len(structures), len(all_species)] with compositional features
    '''
    result = np.zeros([len(structures), len(all_species)])
    for i, structure in tqdm.tqdm(enumerate(structures)):
        species_now = structure.get_atomic_numbers()
        for j, specie in enumerate(all_species):
            num = np.sum(species_now == specie)
            result[i, j] = num
    return result


def get_spherical_expansion(structures,
                            rascal_hypers,
                            all_species,
                            task_size=100,
                            num_threads=None,
                            show_progress=True):
    '''getting spherical expansion coefficients
    
    Args:
        structures: list of Ase atoms objects        
        rascal_hypers: dictionary with parameters for librascal controlling spherical expansion        
        all_species: numpy array with ints of all unique species in the dataset. \
        If all species argument is the same for several calls of this function, resulting \
        blocks of spherical expansion coefficients are guaranteed to be consisted with each other
        task_size: number of structures in chunk for multiprocessing
        num_threads: number of threads in multiprocessing. If None than all available \
        (len(os.sched_getaffinity(0))) threads are used
        show_progress: whether or not show progress via tqdm
       
    Return:
        dictionary in which keys are elements of all_speceis and entries are numpy arrays with indexing
        [environmental index, radial basis/neighbor specie index, lambda, m] with spherical expansion coefficients for 
        environments around atoms with specie indicated in key. Coefficients are stored from the beginning,
        i. e. [:, : lambda, :(2 * lambda + 1)] elements are valid
    '''
    hypers = copy.deepcopy(rascal_hypers)

    if ('expansion_by_species_method' in hypers.keys()):
        if (hypers['expansion_by_species_method'] != 'user defined'):
            raise ValueError(
                "for proper packing spherical expansion coefficients into [env index, radial/specie index, l, m] shape output should be uniform, thus 'expansion_by_species_method' must be 'user defined'"
            )

    hypers['expansion_by_species_method'] = 'user defined'

    species_list = []
    for structure in structures:
        species_list.append(structure.get_atomic_numbers())
    species_list = np.concatenate(species_list, axis=0)
    species_list = species_list.astype(np.int32)
    all_species = all_species.astype(np.int32)

    if ('global_species' not in hypers.keys()):
        hypers['global_species'] = [int(specie) for specie in all_species]
    else:
        for specie in all_species:
            if (specie not in hypers['global_species']):
                warnings.warn(
                    "atom with type {} is presented in the all_species argument to this function but it is not listed in the global_species, adding it"
                    .format(specie))
                hypers['global_species'].append(int(specie))

        all_species = np.array(hypers['global_species']).astype(np.int32)

    if (num_threads is None):
        num_threads = len(os.sched_getaffinity(0))

    p = Pool(num_threads)
    tasks = []
    for i in range(0, len(structures), task_size):
        tasks.append([structures[i:i + task_size], hypers, len(all_species)])

    result = [
        res for res in tqdm.tqdm(p.imap(
            nice.rascal_coefficients.get_rascal_coefficients_stared, tasks),
                                 total=len(tasks),
                                 disable=not show_progress)
    ]
    p.close()
    p.join()
    result = np.concatenate(result, axis=0)
    return nice.rascal_coefficients.split_by_central_specie(
        species_list, all_species, result, show_progress=show_progress)


def make_structural_features(features,
                             structures,
                             all_species,
                             show_progress=True):
    ''' getting structural features suitable for linear regression which consist of sums \
     over atomic features
    
    Args:
        features: nested dictionary with atomic features. First level keys are central species, \
        second level keys are body orders. Entries are 2-dimensional numpy arrays. 
        structures: list of Ase atoms objects
        all_species: numpy array with ints of all unique species in the dataset. \
        If all species argument is the same for several calls of this function, resulting \
        blocks of structural features are guaranteed to be consistent with each other.   \
        If for given block of structures there are no atoms of some particular specie,\
        features dictionary still have to contain key with this specie. It should contain \
        numpy arrays with shapes [0, number of features]. This is need to get proper placing\
        of features to fulfill consistency.
        show_progress: whether or not show progress via tqdm
        
    Return:
        numpy array with shape [len(structures), number of structural features] with structural features
    '''

    for specie in all_species:
        if (specie not in features.keys()):
            raise ValueError(
                "all_species contains atomic specie {}, "
                "but there are no features for it. "
                "In case of absence of such atoms in given set "
                "of structures provide empty array with shape "
                "[0, num_features] which is needed to "
                "determine proper shape of output ".format(specie))

    start_indices, end_indices = {}, {}
    now = 0
    for specie_index in all_species:
        start_indices[specie_index] = {}
        end_indices[specie_index] = {}
        for body_order_index in features[specie_index].keys():
            start_indices[specie_index][body_order_index] = now
            now += features[specie_index][body_order_index].shape[1]
            end_indices[specie_index][body_order_index] = now

    total_size = now

    result = np.zeros([len(structures), total_size])

    current_positions = {}
    for specie in all_species:
        current_positions[specie] = 0

    for i in tqdm.tqdm(range(len(structures)), disable=not (show_progress)):
        species_now = structures[i].get_atomic_numbers()
        for specie in all_species:
            num_atoms_now = np.sum(species_now == specie)
            if (num_atoms_now == 0):
                continue

            for body_order in features[specie].keys():
                features_now = np.sum(
                    features[specie][body_order][current_positions[specie]:(
                        current_positions[specie] + num_atoms_now)],
                    axis=0)
                result[i, start_indices[specie][body_order]:end_indices[specie]
                       [body_order]] = features_now

            current_positions[specie] += num_atoms_now

    return result


def transform_sequentially(nice,
                           structures,
                           rascal_hypers,
                           all_species,
                           block_size=500,
                           show_progress=True):
    ''' transforming structures into structural features by chunks in order to use less amount of RAM
    
    Args:
        nice: dictionary where keys are species and entries are nice transformers.\
        If you want to use single nice transformer to all environments regardless of central\
        specie just pass {key : nice_single for specie in all_species}
        structures: list of Ase atoms objects
        rascal_hypers: dictionary with parameters for librascal controlling spherical expansion.\
        Should be the same as used for fitting nice transformers
        all_species: numpy array with ints of all unique species in the dataset. 
        block_size: size of chunks measured in number of environments
        show_progress: whether or not show progress via tqdm
        
        
    Return:
        numpy array with shape [len(structures), number of structural features] with structural features
    '''

    pieces = []

    for i in tqdm.tqdm(range(0, len(structures), block_size),
                       disable=not show_progress):
        now = {}
        coefficients = get_spherical_expansion(structures[i:i + block_size],
                                               rascal_hypers,
                                               all_species,
                                               show_progress=False)
        for specie in all_species:
            if (coefficients[specie].shape[0] != 0):
                now[specie] = nice[specie].transform(
                    coefficients[specie], return_only_invariants=True)
            else:
                # determining size of output
                dummy_shape = coefficients[specie].shape
                dummy_shape = list(dummy_shape)
                dummy_shape[0] = 1
                dummy_data = np.ones(dummy_shape)
                dummy_output = nice[specie].transform(
                    dummy_data, return_only_invariants=True)
                current_block = {}
                for key in dummy_output.keys():
                    current_block[key] = np.zeros(
                        [0, dummy_output[key].shape[1]])
                now[specie] = current_block

        pieces.append(
            make_structural_features(now,
                                     structures[i:i + block_size],
                                     all_species,
                                     show_progress=False))

    return np.concatenate(pieces, axis=0)
