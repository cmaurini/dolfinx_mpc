import typing

import numba
from petsc4py import PETSc
import types
from dolfinx import function, fem, MPI
from .assemble_matrix import in_numpy_array
import numpy


def backsubstitution(mpc, vector, dofmap):
    slaves = mpc.slaves()
    masters, coefficients = mpc.masters_and_coefficients()
    offsets = mpc.master_offsets()
    index_map = mpc.index_map()
    slave_cells = mpc.slave_cells()
    cell_to_slave, cell_to_slave_offset = mpc.cell_to_slave_mapping()
    ghost_info = (index_map.local_range, index_map.ghosts,
                  index_map.indices(True))
    mpc = (slaves, slave_cells, cell_to_slave, cell_to_slave_offset,
           masters, coefficients, offsets)

    backsubstitution_numba(vector, dofmap.dof_array, mpc, ghost_info)
    vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
    return vector


@numba.njit
def backsubstitution_numba(b, dofmap, mpc, ghost_info):
    """
        Insert mpc values into vector bc
        """
    (slaves, slave_cells, cell_to_slave, cell_to_slave_offset,
     masters, coefficients, offsets) = mpc
    (local_range, ghosts, global_indices) = ghost_info
    slaves_visited = numpy.empty(0, dtype=numpy.float64)
    # Loop through slave cells
    for (index, cell_index) in enumerate(slave_cells):
        cell_slaves = cell_to_slave[cell_to_slave_offset[index]:
                                    cell_to_slave_offset[index+1]]
        local_dofs = dofmap[3 * cell_index:3 * cell_index + 3]

        # Find the global index of the slaves on the cell in the slaves-array
        global_slaves_index = []
        for gi in range(len(slaves)):
            if in_numpy_array(cell_slaves, slaves[gi]):
                global_slaves_index.append(gi)

        for slave_index in global_slaves_index:
            slave = slaves[slave_index]
            k = -1
            # Find local position of slave dof
            for local_dof in local_dofs:
                if global_indices[local_dof] == slave:
                    k = local_dof
            assert k != -1
            # Check if we have already inserted for this slave
            if not in_numpy_array(slaves_visited, slave):
                slaves_visited = numpy.append(slaves_visited, slave)
                slaves_masters = masters[offsets[slave_index]:
                                         offsets[slave_index+1]]
                slaves_coeffs = coefficients[offsets[slave_index]:
                                             offsets[slave_index+1]]
                for (master, coeff) in zip(slaves_masters, slaves_coeffs):
                    # Find local index for master
                    local_master_index = -1
                    if master < local_range[1] and master >= local_range[0]:
                        local_master_index = master-local_range[0]
                    else:
                        for q, ghost in enumerate(ghosts):
                            if master == ghost:
                                local_master_index = q + \
                                    local_range[1] - local_range[0]
                                break
                    assert q != -1
                    b[k] += coeff*b[local_master_index]


def slave_master_structure(V: function.FunctionSpace, slave_master_dict:
                           typing.Dict[types.FunctionType,
                                       typing.Dict[
                                           types.FunctionType, float]]):
    """
    Returns the data structures required to build a multi-point constraint.
    Given a nested dictionary, where the first keys are functions for
    geometrically locating the slave degrees of freedom. The values of these
    keys are another dictionary, containing functions for geometrically
    locating the master degree of freedom. The value of the nested dictionary
    is the coefficient the master degree of freedom should be multiplied with
    in the multi point constraint.
    Example:
       If u0 = alpha u1 + beta u2, u3 = beta u4 + gamma u5
       slave_master_dict = {lambda x loc_u0:{lambda x loc_u1: alpha,
                                             lambda x loc_u2: beta},
                            lambda x loc_u3:{lambda x loc_u4: beta,
                                             lambda x loc_u5: gamma}}
    """
    slaves = []
    masters = []
    coeffs = []
    offsets = []
    local_min = V.dofmap.index_map.local_range[0]
    for slave in slave_master_dict.keys():
        offsets.append(len(masters))

        dof = fem.locate_dofs_geometrical(V, slave) + local_min
        dof_global = numpy.vstack(MPI.comm_world.allgather(dof))[0]
        slaves.append(dof_global)
        for master in slave_master_dict[slave].keys():
            dof_m = fem.locate_dofs_geometrical(V, master) + local_min
            dof_m = numpy.vstack(MPI.comm_world.allgather(dof_m))[0]
            masters.append(dof_m)
            coeffs.append(slave_master_dict[slave][master])
    offsets.append(len(masters))
    return (numpy.array(slaves), numpy.array(masters),
            numpy.array(coeffs, dtype=numpy.float64), numpy.array(offsets))


def dof_close_to(x, point):
    """
    Convenience function for locating a dof close to a point use numpy
    and lambda functions.
    """
    if point is None:
        raise ValueError("Point must be supplied")
    if len(point) == 1:
        return numpy.isclose(x[0], point[0])
    elif len(point) == 2:
        return numpy.logical_and(numpy.isclose(x[0], point[0]),
                                 numpy.isclose(x[1], point[1]))
    elif len(point) == 3:
        return numpy.logical_and(
            numpy.logical_and(numpy.isclose(x[0], point[0]),
                              numpy.isclose(x[1], point[1]),
                              numpy.isclose(x[2], point[2])))
    else:
        return ValueError("Point has to be 1D, 2D or 3D")
