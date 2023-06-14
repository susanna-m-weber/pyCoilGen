import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilSolution, BasisElement, CoilPart
from sub_functions.gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle

log = logging.getLogger(__name__)


def calculate_sensitivity_matrix(coil_solution: CoilSolution, coil_parts: List[CoilPart], target_field, input):
    """
    Calculate the sensitivity matrix.

    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        target_field: The target field.
        optimisation: Optimisation parameters.

    Returns:
        List[CoilPart]: Updated list of coil parts with sensitivity matrix.

    """
    optimisation = coil_solution.optimisation  # Retrieve the solution optimization parameters

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part

        if not optimisation.use_preoptimization_temp:
            target_points = target_field.coords
            gauss_order = input.gauss_order

            # Calculate the weights and the test point for the Gauss-Legendre integration on each triangle
            u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)
            num_gauss_points = len(gauss_weight)
            biot_savart_coeff = 1e-7
            num_nodes = len(coil_part.basis_elements)
            num_target_points = target_points.shape[1]
            sensitivity_matrix = np.zeros((3, num_target_points, num_nodes))

            for node_ind in range(num_nodes):
                dCx = np.zeros(num_target_points)
                dCy = np.zeros(num_target_points)
                dCz = np.zeros(num_target_points)

                for tri_ind in range(len(coil_part.basis_elements[node_ind].area)):
                    node_point = coil_part.basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 0]
                    point_b = coil_part.basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 1]
                    point_c = coil_part.basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 2]

                    x1, y1, z1 = node_point
                    x2, y2, z2 = point_b
                    x3, y3, z3 = point_c

                    vx, vy, vz = coil_part.basis_elements[node_ind].current[tri_ind]

                    for gauss_ind in range(num_gauss_points):
                        xgauss_in_uv = x1 * u_coord[gauss_ind] + x2 * v_coord[gauss_ind] + x3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                        ygauss_in_uv = y1 * u_coord[gauss_ind] + y2 * v_coord[gauss_ind] + y3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                        zgauss_in_uv = z1 * u_coord[gauss_ind] + z2 * v_coord[gauss_ind] + z3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])

                        distance_norm = ((xgauss_in_uv - target_points[0])**2 + (ygauss_in_uv - target_points[1])**2 + (zgauss_in_uv - target_points[2])**2)**(-3/2)

                        dCx += ((-1) * vz * (target_points[1] - ygauss_in_uv) + vy * (target_points[2] - zgauss_in_uv)) * distance_norm * 2 * coil_part.basis_elements[node_ind].area[tri_ind] * gauss_weight[gauss_ind]
                        dCy += ((-1) * vx * (target_points[2] - zgauss_in_uv) + vz * (target_points[0] - xgauss_in_uv)) * distance_norm * 2 * coil_part.basis_elements[node_ind].area[tri_ind] * gauss_weight[gauss_ind]
                        dCz += ((-1) * vy * (target_points[0] - xgauss_in_uv) + vx * (target_points[1] - ygauss_in_uv)) * distance_norm * 2 * coil_part.basis_elements[node_ind].area[tri_ind] * gauss_weight[gauss_ind]

                sensitivity_matrix[:, :, node_ind] = np.array([dCx, dCy, dCz]) * biot_savart_coeff

            coil_part.sensitivity_matrix = sensitivity_matrix

        else:
            coil_part.sensitivity_matrix = optimisation.temp.coil_parts[part_ind].sensitivity_matrix

    return coil_parts
