import tqdm
import numpy as np


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

    start_indices, end_indices = {}
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

    for i in tqdm.tqdm(range(len(structures), disable=not (show_progress))):
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
