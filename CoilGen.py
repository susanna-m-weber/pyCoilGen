# System imports
import numpy as np

# Logging
import logging

from sub_functions.data_structures import Mesh

# Local imports
from sub_functions.constants import *
from sub_functions.data_structures import DataStructure, CoilSolution, OptimisationParameters

# For visualisation
from helpers.visualisation import visualize_vertex_connections, visualize_compare_contours

# For timing
from helpers.timing import Timing

# From original project
from sub_functions.read_mesh import read_mesh
from sub_functions.parse_input import parse_input, create_input
from sub_functions.split_disconnected_mesh import split_disconnected_mesh
from sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
# from sub_functions.refine_mesh import refine_mesh # Broken
from sub_functions.parameterize_mesh import parameterize_mesh
from sub_functions.define_target_field import define_target_field
# from sub_functions.temp_evaluation import temp_evaluation
from sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh
from sub_functions.calculate_basis_functions import calculate_basis_functions
from sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
from sub_functions.calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from sub_functions.calculate_resistance_matrix import calculate_resistance_matrix
from sub_functions.stream_function_optimization import stream_function_optimization
from sub_functions.calc_potential_levels import calc_potential_levels
from sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts
from sub_functions.process_raw_loops import process_raw_loops
from sub_functions.find_minimal_contour_distance import find_minimal_contour_distance
from sub_functions.topological_loop_grouping import topological_loop_grouping
from sub_functions.calculate_group_centers import calculate_group_centers
from sub_functions.interconnect_within_groups import interconnect_within_groups
from sub_functions.interconnect_among_groups import interconnect_among_groups
from sub_functions.shift_return_paths import shift_return_paths
from sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
from sub_functions.create_sweep_along_surface import create_sweep_along_surface
from sub_functions.calculate_inductance_by_coil_layout import calculate_inductance_by_coil_layout
"""
from evaluate_field_errors import evaluate_field_errors
from calculate_gradient import calculate_gradient
from load_preoptimized_data import load_preoptimized_data
"""

# Set up logging
log = logging.getLogger(__name__)


def save(output_dir, project_name, tag, solution):
    filename = f'{output_dir}/{project_name}_{tag}.npy'
    if get_level() > DEBUG_NONE:
        log.debug("Saving solution to '%s'", filename)
    np.save(filename, np.asarray([solution], dtype=object))


def CoilGen(log, input_args=None):
    # Create optimized coil finished coil layout
    # Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology, Medical Physics
    # 5.10.2021

    # The following external functions were used in modified form:
    # intreparc@John D'Errico (2010), @matlabcentral/fileexchange
    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan Schmidt rms@dgp.toronto.edu"
    # based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes", NS (2021).
    # Curve intersections (https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections),
    # MATLAB Central File Exchange.

    timer = Timing()
    timer.start()

    # Parse the input variables
    if type(input_args) is dict:
        if input_args['debug'] >= DEBUG_VERBOSE:
            log.debug(" - converting input dict to input type.")
        input_parser, input_args = create_input(input_args)
    elif input_args is None:
        input_parser, input_args = parse_input(input_args)
    else:
        input_args = input_args

    set_level(input_args.debug)

    project_name = input_args.project_name
    persistence_dir = input_args.persistence_dir

    # Print the input variables
    # DEBUG
    if get_level() >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)

    solution = CoilSolution()
    solution.input_args = input_args

    if input_args.sf_source_file == 'none':
        # Read the input mesh
        print('Load geometry:')
        coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)  # 01

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- vertices shape: %s", coil_mesh.get_vertices().shape)  # (264,3)
            log.debug(" -- faces shape: %s", coil_mesh.get_faces().shape)  # (480,3)

        if get_level() > DEBUG_VERBOSE:
            log.debug(" coil_mesh.vertex_faces: %s", coil_mesh.trimesh_obj.vertex_faces[0:10])

        if get_level() > DEBUG_VERBOSE:
            coil_mesh.display()

        # Split the mesh and the stream function into disconnected pieces
        print('Split the mesh and the stream function into disconnected pieces.')
        timer.start()
        coil_parts = split_disconnected_mesh(coil_mesh)  # 00
        timer.stop()
        solution.coil_parts = coil_parts
        save(persistence_dir, project_name, '00', solution)

        # Upsample the mesh density by subdivision
        print('Upsample the mesh by subdivision:')
        timer.start()
        coil_parts = refine_mesh(coil_parts, input_args)  # 01
        timer.stop()
        save(persistence_dir, project_name, '01', solution)

        # Parameterize the mesh
        print('Parameterize the mesh:')
        timer.start()
        coil_parts = parameterize_mesh(coil_parts, input_args)  # 02
        timer.stop()
        save(persistence_dir, project_name, '02', solution)

        # Define the target field
        print('Define the target field:')
        timer.start()
        target_field, is_suppressed_point = define_target_field(
            coil_parts, target_mesh, secondary_target_mesh, input_args)
        timer.stop()
        solution.target_field = target_field
        solution.is_suppressed_point = is_suppressed_point
        save(persistence_dir, project_name, '02b', solution)

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- target_field.b shape: %s", target_field.b.shape)  # (3, 257)
            log.debug(" -- target_field.coords shape: %s", target_field.coords.shape)  # (3, 257)
            log.debug(" -- target_field.weights shape: %s", target_field.weights.shape)  # (257,)

        # Evaluate the temp data; check whether precalculated values can be used from previous iterations
        # print('Evaluate the temp data:')
        # input_args = temp_evaluation(solution, input_args, target_field)

        # Find indices of mesh nodes for one ring basis functions
        print('Calculate mesh one ring:')
        timer.start()
        coil_parts = calculate_one_ring_by_mesh(coil_parts)  # 03
        timer.stop()
        save(persistence_dir, project_name, '03', solution)

        # Create the basis function container which represents the current density
        print('Create the basis function container which represents the current density:')
        timer.start()
        coil_parts = calculate_basis_functions(coil_parts)  # 04
        timer.stop()
        save(persistence_dir, project_name, '04', solution)

        # Calculate the sensitivity matrix Cn
        print('Calculate the sensitivity matrix:')
        timer.start()
        coil_parts = calculate_sensitivity_matrix(coil_parts, target_field, input_args)  # 05
        timer.stop()
        save(persistence_dir, project_name, '05', solution)

        # Calculate the gradient sensitivity matrix Gn
        print('Calculate the gradient sensitivity matrix:')
        timer.start()
        coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)  # 06
        timer.stop()
        save(persistence_dir, project_name, '06', solution)

        # Calculate the resistance matrix Rmn
        print('Calculate the resistance matrix:')
        timer.start()
        coil_parts = calculate_resistance_matrix(coil_parts, input_args)  # 07
        timer.stop()
        save(persistence_dir, project_name, '07', solution)

        # Optimize the stream function toward target field and further constraints
        print('Optimize the stream function toward target field and secondary constraints:')
        timer.start()
        coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)  # 08
        timer.stop()
        solution.combined_mesh = combined_mesh
        solution.sf_b_field = sf_b_field
        save(persistence_dir, project_name, '08', solution)

    else:
        # Load the preoptimized data
        print('Load preoptimized data:')
        raise Exception("Not supported")
        timer.start()
        solution = load_preoptimized_data(input_args)
        timer.stop()
        #coil_parts, _, _, combined_mesh, sf_b_field, target_field, is_suppressed_point = 
        # load_preoptimized_data(input_args)

    # Calculate the potential levels for the discretization
    print('Calculate the potential levels for the discretization:')
    timer.start()
    coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)  # 09
    timer.stop()
    solution.primary_surface_ind = primary_surface_ind
    save(persistence_dir, project_name, '09', solution)

    # Generate the contours
    print('Generate the contours:')
    timer.start()
    coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)  # 10
    timer.stop()
    save(persistence_dir, project_name, '10', solution)

    # Process contours
    print('Process contours: Evaluate loop significance')
    timer.start()
    coil_parts = process_raw_loops(coil_parts, input_args, target_field)  # 11
    timer.stop()
    save(persistence_dir, project_name, '11', solution)

    if not input_args.skip_postprocessing:
        # Find the minimal distance between the contour lines
        print('Find the minimal distance between the contour lines:')
        timer.start()
        coil_parts = find_minimal_contour_distance(coil_parts, input_args)  # 12
        timer.stop()
        save(persistence_dir, project_name, '12', solution)

        # Group the contour loops in topological order
        print('Group the contour loops in topological order:')
        timer.start()
        coil_parts = topological_loop_grouping(coil_parts, input_args)  # 13
        timer.stop()
        save(persistence_dir, project_name, '13', solution)

        # Calculate center locations of groups
        print('Calculate center locations of groups:')
        timer.start()
        coil_parts = calculate_group_centers(coil_parts)  # 14
        timer.stop()
        save(persistence_dir, project_name, '14', solution)

        #####################################################
        # Visualisation
        if get_level() > DEBUG_NONE:
            for part_index in range(len(coil_parts)):
                coil_part = coil_parts[part_index]
                coil_mesh = coil_part.coil_mesh
                c_group_centers = coil_part.group_centers

                visualize_compare_contours(coil_mesh.uv, 800, f'images/14_{project_name}_contour_centres_{part_index}_p.png',
                                           coil_part.contour_lines, c_group_centers.uv)
        #
        #####################################################

        # Interconnect the single groups
        print('Interconnect the single groups:')
        timer.start()
        coil_parts = interconnect_within_groups(coil_parts, input_args)  # 15
        timer.stop()
        save(persistence_dir, project_name, '15', solution)

        # Interconnect the groups to a single wire path
        print('Interconnect the groups to a single wire path:')
        timer.start()
        coil_parts = interconnect_among_groups(coil_parts, input_args)  # 16
        timer.stop()
        save(persistence_dir, project_name, '16', solution)

        # Connect the groups and shift the return paths over the surface
        print('Shift the return paths over the surface:')
        timer.start()
        coil_parts = shift_return_paths(coil_parts, input_args)  # 17
        timer.stop()
        save(persistence_dir, project_name, '17', solution)

        #####################################################
        # Visualisation
        if get_level() > DEBUG_NONE:
            for index1 in range(len(coil_parts)):
                c_part = coil_parts[index1]
                c_wire_path = c_part.wire_path

                visualize_vertex_connections(
                    c_wire_path.uv.T, 800, f'images/17_{project_name}_wire_path2_uv_{index1}_p.png')
        #
        #####################################################

        # Create Cylindrical PCB Print
        print('Create PCB Print:')
        timer.start()
        coil_parts = generate_cylindrical_pcb_print(coil_parts, input_args)  # 18
        timer.stop()
        save(persistence_dir, project_name, '18', solution)

        # Create Sweep Along Surface
        print('Create sweep along surface:')
        timer.start()
        coil_parts = create_sweep_along_surface(coil_parts, input_args)
        timer.stop()
        save(persistence_dir, project_name, '19', solution)

    # Calculate the inductance by coil layout
    print('Calculate the inductance by coil layout:')
    # coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance 
    timer.start()
    solution = calculate_inductance_by_coil_layout(solution, input_args)
    timer.stop()

    # WIP
    timer.stop()
    return solution

    # Evaluate the field errors
    print('Evaluate the field errors:')
    field_errors, _, _ = evaluate_field_errors(coil_parts, target_field, input_args)

    # Calculate the gradient
    print('Calculate the gradient:')
    coil_gradient = calculate_gradient(coil_parts, target_field, input_args)

    return solution


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    # create cylinder mesh: 0.4, 0.1125, 50, 50, copy from Matlab

    # Examples/biplanar_xgradient.m
    arg_dict1 = {
        #"area_perimeter_deletion_ratio": 5,
        #"b_0_direction": [0, 0, 1],
        #"biplanar_mesh_parameter_list": [0.25, 0.25, 20, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        #"circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        #"circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "bi_planer_rectangles_width_1000mm_distance_500mm.stl",
        #"conductor_cross_section_height": 0.002,
        #"conductor_cross_section_width": 0.002,
        #"conductor_thickness": 0.005,
        #"cross_sectional_points": [0, 0],
        #"cylinder_mesh_parameter_list": [0.4, 0.1125, 50, 50, 0.0, 1.0, 0.0, 0.0],
        #"double_cone_mesh_parameter_list": [0.8, 0.3, 0.3, 0.1, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "x",
        "fieldtype_to_evaluate": ['', 'MCOS', 'string', [3707764736,          2,          1,          1,          2,                2]],
        "fmincon_parameter": [500.0, 10000000000.0, 1e-10, 1e-10, 1e-10],
        "force_cut_selection": ['high'],
        #"gauss_order": 2,
        #"group_interconnection_method": "crossed",
        "interconnection_cut_width": 0.05,
        "interconnection_method": "regular",
        "iteration_num_mesh_refinement": 1,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 14,
        #"make_cylindrical_pcb": 0,
        #"max_allowed_angle_within_coil_track": 120,
        #"min_allowed_angle_within_coil_track": 0.0001,
        #"min_loop_significance": 1,
        #"min_point_loop_number": 20,
        "normal_shift_length": 0.01,
        #"normal_shift_smooth_factors": [2, 3, 2],
        "output_directory": "images",
        #"pcb_interconnection_method": "spiral_in_out",
        #"pcb_spiral_end_shift_factor": 10,
        #"planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #"plot_flag": 1,
        "pot_offset_factor": 0.25,
        #"save_stl_flag": True,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        #"sf_opt_method": "tikkonov",
        "sf_source_file": "none",
        #"skip_calculation_min_winding_distance": 1,  # Default: 1
        "skip_inductance_calculation": False,
        #"skip_normal_shift": 0,
        "skip_postprocessing": False,
        #"skip_sweep": 0,
        #"smooth_factor": 1,
        #"smooth_flag": 1,
        #"specific_conductivity_conductor": 1.8e-8,
        "surface_is_cylinder_flag": 1,
        #"target_field_definition_field_name": "none",
        #"target_field_definition_file": "none",
        #"target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.1,    # GitHub
        #"target_region_resolution": 10, # MATLAB 10 is the default but 5 is faster
        "tikonov_reg_factor": 10,
        #"tiny_segment_length_percentage": 0,
        #"track_width_factor": 0.5,
        "use_only_target_mesh_verts": False,
        "project_name": 'biplanar_xgradient',
        "fasthenry_bin": '../FastHenry2/bin/fasthenry',
        "persistence_dir": 'debug',
        "debug": DEBUG_BASIC,
    }  # 4m3, 6m12.747s

    # cylinder_radius500mm_length1500mm
    arg_dict2 = {
        "area_perimeter_deletion_ratio": 5,
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        "circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "cylinder_radius500mm_length1500mm.stl",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.015,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18], [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]],
        "cylinder_mesh_parameter_list": [0.8, 0.3, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "double_cone_mesh_parameter_list": [0.8, 0.3, 0.3, 0.1, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "y",
        "fieldtype_to_evaluate": ['', 'MCOS', 'string', [3707764736, 2, 1, 1, 1, 1]],
        "fmincon_parameter": [1000.0, 10000000000.0, 1e-10, 1e-10, 1e-10],
        "force_cut_selection": ['high'],
        "gauss_order": 2,
        # "geometry_source_path": "/MATLAB Drive/CoilGen/Geometry_Data",
        "group_interconnection_method": "crossed",
        "interconnection_cut_width": 0.1,
        "interconnection_method": "regular",
        "iteration_num_mesh_refinement": 1,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 20,
        "make_cylindrical_pcb": 1,
        "max_allowed_angle_within_coil_track": 120,
        "min_allowed_angle_within_coil_track": 0.0001,
        "min_loop_significance": 0.1,
        "min_point_loop_number": 20,
        "normal_shift_length": 0.025,
        "normal_shift_smooth_factors": [2, 3, 2],
        "output_directory": "images",
        "pcb_interconnection_method": "spiral_in_out",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plot_flag": 1,
        "pot_offset_factor": 0.25,
        "project_name": 'cylinder',
        "save_stl_flag": 1,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        "sf_opt_method": "tikkonov",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": 1,  # Default 1
        "skip_inductance_calculation": 0,
        "skip_normal_shift": 0,
        "skip_postprocessing": 0,
        "skip_sweep": 0,
        "smooth_factor": 1,
        "smooth_flag": 1,
        "specific_conductivity_conductor": 1.8e-08,
        "surface_is_cylinder_flag": True,
        "target_field_definition_field_name": "none",
        "target_field_definition_file": "none",
        "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.15,
        "target_region_resolution": 10,  # MATLAB 10 is the default but 5 is faster
        "tikonov_reg_factor": 100,
        "tiny_segment_length_percentage": 0,
        "track_width_factor": 0.5,
        "use_only_target_mesh_verts": False,
        "debug": DEBUG_BASIC,
        "persistence_dir": 'debug',
        "fasthenry_bin": '../FastHenry2/bin/fasthenry',
    }  # 2m11

    solution = CoilGen(log, arg_dict1)
