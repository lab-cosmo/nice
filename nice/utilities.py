import tqdm
import numpy as np
import nice.rascal_coefficients

def make_structural_features(features, structures, show_progress=True):
    all_species = []
    for structure in structures:
        all_species.append(np.array(structure.get_atomic_numbers()))
    all_species = np.concatenate(all_species, axis=0)

    all_species = np.sort(np.unique(all_species))

    for specie in all_species:
        if (specie not in features.keys()):
            raise ValueError("structures contain atomic specie {}, "
                             "but not features given for it".format(specie))

    start_indices, end_indices = {}, {}
    now = 0
    for specie_index in features.keys():
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

def get_spherical_expansion(structures, rascal_hypers, task_size = 100, num_threads = None, show_progress = True):  
    hypers = copy.deepcopy(rascal_hypers)
    
    if ('expansion_by_species_method' in hypers.keys()):
        if (hypers['expansion_by_species_method'] != 'user defined'):
            raise ValueError("for proper packing spherical expansion coefficients into [env index, radial/specie index, l, m] shape output should be uniform, thus 'expansion_by_species_method' must be 'user defined'")
            
    hypers['expansion_by_species_method'] = 'user defined'
    
    
    all_species = []
    for structure in structures:
        all_species.append(structure.get_atomic_numbers())
    all_species = np.concatenate(all_species, axis = 0)
    species = np.unique(all_species)
    all_species = all_species.astype(np.int32)
    species = species.astype(np.int32)
    if ('global_species' not in hypers.keys()):
        hypers['global_species'] = [int(specie) for specie in species]
    else:
        for specie in species:
            if (specie not in hypers['global_species']):
                warnings.warn("atom with type {} is presented in the dataset but it is not listed in the global_species, adding it".format(specie))
                hypers['global_species'].append(int(specie))
            
        species = np.array(hypers['global_species']).astype(np.int32)
                
      
                              
    
    if (num_threads is None):
        num_threads = cpu_count()
    
    p = Pool(num_threads) 
    tasks = []
    for i in range(0, len(structures), task_size):
        tasks.append([structures[i : i + task_size], hypers, len(species)])  
        
    result = [res for res in tqdm.tqdm(p.imap(get_rascal_coefficients_stared, tasks), total = len(tasks), disable = not show_progress)]
    p.close()
    p.join()
    result = np.concatenate(result, axis = 0)
    return split_by_central_specie(all_species, species, result, show_progress = show_progress)
