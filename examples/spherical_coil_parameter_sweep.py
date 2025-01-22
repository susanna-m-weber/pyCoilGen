# System imports
import itertools
import numpy as np
import multiprocessing
from os import makedirs

# Logging
import logging


# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
from pyCoilGen.plotting import plot_error_different_solutions

"""
Author: Kevin Meyer
Bela Pena s.p.
September 2023

Demonstrate multiprocessing to generate multiple solutions in order to sweep a parameter set.

Since the processing thread can terminate without notice, e.g. when out of resources, run this example multiple times
until all solutions have been generated.
"""


def project_name(param_dict, combination):
    """Compute a project name based on the swept parameters"""
    # Create unique project name out of swept parameters
    project_name = param_dict['project_name']
    suffix = ''
    for x in combination:
        suffix += f'_{x}'
    return project_name+suffix


def process_combination(combination):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    constant_params = {
        'field_shape_function': '100.0',  # definition of the target field
        'target_gradient_strength': 1,
        'coil_mesh_file': 'flattened_sphere_9.stl', 
        'target_region_radius': 0.25,  # in meter
        'sf_source_file': 'none',
        'surface_is_cylinder_flag': False,
        'set_roi_into_mesh_center': True,
        'use_only_target_mesh_verts': False,
        'skip_normal_shift': False,
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'force_cut_selection': ['high'],
        'level_set_method': 'primary',  # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
 
        'levels': 17,
        'tikhonov_reg_factor': 3,
        'pot_offset_factor': 0.2,  # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'interconnection_cut_width': 0.005,  # the width for the interconnections are interconnected; in meter
       
        'conductor_thickness': 0.07,
        #'normal_shift_length': 0.01,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
       
        
        'output_directory': 'images',  # [Current directory]
        'project_name': 'flattened_sphere_9_r4',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    sweep_params = {
        'normal_shift_length': [0.0005, 0.001, 0.002, 0.004, 0.004, 0.005],  

        'interconnection_cut_width': [0.00005, 0.001, 0.002, 0.003, 0.004, 0.005]
    }

    # Create a copy of the constant parameters
    param_dict = constant_params.copy()

    # Merge in the sweep parameters
    param_dict.update({param_name: param_value for param_name, param_value in zip(sweep_params.keys(), combination)})

    # Update the project name to reflect the current combination
    param_dict['project_name'] = project_name(param_dict, combination)
    log.info('Starting %s', param_dict['project_name'])

    # Calculate the result
    result = pyCoilGen(log, param_dict)
    return result


if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    pcb_width = 0.002
    cut_width = 0.025
    normal_shift = 0.005
    min_loop_significance = 3

    constant_params = {
        'field_shape_function': '100.0',  # definition of the target field
        'target_gradient_strength': 1,
        'coil_mesh_file': 'flattened_sphere_9.stl', 
        'target_region_radius': 0.25,  # in meter
        'sf_source_file': 'none',
        'surface_is_cylinder_flag': False,
        'set_roi_into_mesh_center': True,
        'use_only_target_mesh_verts': False,
        'skip_normal_shift': False,
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'force_cut_selection': ['high'],
        'level_set_method': 'primary',  # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
 
        'levels': 17,
        'tikhonov_reg_factor': 3,
        'pot_offset_factor': 0.2,  # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'interconnection_cut_width': 0.005,  # the width for the interconnections are interconnected; in meter
       
        'conductor_thickness': 0.07,
        #'normal_shift_length': 0.01,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
       
        
        'output_directory': 'images',  # [Current directory]
        'project_name': 'flattened_sphere_9_r4',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    sweep_params = {
        'normal_shift_length': [0.0005, 0.001, 0.002, 0.004, 0.004, 0.005],  

        'interconnection_cut_width': [0.00005, 0.001, 0.002, 0.003, 0.004, 0.005]
    }


    # Generate all combinations of parameters
    parameter_combinations = itertools.product(*sweep_params.values())

    # Check if outputs already exist, try and load all combinations:
    # Might need to repeat this multiple times in case one process terminates unexpectedly.
    missing = []
    results = []
    for combination in parameter_combinations:
        try:
            project_name_str = project_name(constant_params, combination)
            file_name = f"{constant_params['persistence_dir']}/{project_name_str}_final.npy"
            [solution] = np.load(file_name, allow_pickle=True)
            log.info("Loaded %s", project_name_str)
            results.append(solution)
        except FileNotFoundError as e:
            missing.append(combination)
            log.warning("File not found: %s", e)

    # If any outputs are missing, create them.
    if len(missing) > 0:
        # Use multiprocessing.Pool to execute solve in parallel
        with multiprocessing.Pool() as pool:
            results = pool.map(process_combination, missing)
    else:
        # results now contains the results of each call to solve
        image_dir = 'images/spherical_coil_param_sweep'
        makedirs(image_dir, exist_ok=True)
        # Plot figures of all levels per tikhonov_reg_factor (i.e. the tikhonov_reg_factor is fixed in each figure)
        for index, tk in enumerate(sweep_params['normal_shift_length']):
            title = f'Circular Coil Study\n(Normal Shift{tk})'
            base = len(sweep_params['interconnection_cut_width'])*index
            to_plot = [base+i for i in range(len(sweep_params['interconnection_cut_width']))]
            plot_error_different_solutions(results, to_plot, title, x_ticks={
                                           'interconnection_cut_width': sweep_params['interconnection_cut_width']}, save_dir=image_dir)
