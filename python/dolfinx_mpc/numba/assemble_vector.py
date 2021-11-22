# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import dolfinx
import dolfinx.common
import dolfinx.log
import numpy
import ufl
from dolfinx_mpc.multipointconstraint import MultiPointConstraint
from petsc4py import PETSc

import numba

from .helpers import extract_slave_cells, pack_slave_facet_info
from .numba_setup import initialize_petsc

Timer = dolfinx.common.Timer

ffi, _ = initialize_petsc()


def assemble_vector(form: ufl.form.Form, constraint: MultiPointConstraint, b: PETSc.Vec = None,
                    form_compiler_parameters={}, jit_parameters={}) -> PETSc.Vec:
    """
    Assemble a linear form into vector b.

    Parameters
    ----------
    form
        The bilinear variational form
    constraint
        The multi point constraint
    b
        PETSc vector to assemble into (optional)
    form_compiler_parameters
        Parameters used in FFCx compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINx.
    jit_parameters
        Parameters used in CFFI JIT compilation of C code generated by FFCx.
        See `python/dolfinx/jit.py` for all available parameters.
        Takes priority over all other parameter values.

    """

    dolfinx.log.log(dolfinx.log.LogLevel.INFO, "Assemble MPC vector")
    timer_vector = Timer("~MPC: Assemble vector (numba)")

    # Unpack Function space data
    V = form.arguments()[0].ufl_function_space()
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x
    dofs = V.dofmap.list.array
    block_size = V.dofmap.index_map_bs

    # Data from multipointconstraint
    coefficients = constraint.coefficients()[0]
    masters_adj = constraint.masters
    c_to_s_adj = constraint.cell_to_slaves
    cell_to_slave = c_to_s_adj.array
    c_to_s_off = c_to_s_adj.offsets
    is_slave = constraint.is_slave
    mpc_data = (masters_adj.array, coefficients, masters_adj.offsets, cell_to_slave, c_to_s_off, is_slave)
    slave_cells = extract_slave_cells(c_to_s_off)

    # Get index map and ghost info
    if b is None:
        index_map = constraint.function_space.dofmap.index_map
        vector = dolfinx.cpp.la.create_vector(index_map, block_size)
    else:
        vector = b

    # If using DOLFINx complex build, scalar type in form_compiler parameters must be updated
    is_complex = numpy.issubdtype(PETSc.ScalarType, numpy.complexfloating)
    if is_complex:
        form_compiler_parameters["scalar_type"] = "double _Complex"

    # Compile ufc form for Python assembly
    ufc_form, _, _ = dolfinx.jit.ffcx_jit(V.mesh.mpi_comm(), form,
                                          form_compiler_parameters=form_compiler_parameters,
                                          jit_parameters=jit_parameters)

    # Pack constants and coefficients
    cpp_form = dolfinx.Form(form, form_compiler_parameters=form_compiler_parameters,
                            jit_parameters=jit_parameters)._cpp_object
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)
    tdim = V.mesh.topology.dim
    num_dofs_per_element = V.dofmap.dof_layout.num_dofs

    # Assemble vector with all entries
    dolfinx.cpp.fem.assemble_vector(vector.array_w, cpp_form, form_consts, form_coeffs)

    # Check if we need facet permutations
    # FIXME: access apply_dof_transformations here
    e0 = cpp_form.function_spaces[0].element
    needs_transformation_data = e0.needs_dof_transformations or cpp_form.needs_facet_permutations
    cell_perms = numpy.array([], dtype=numpy.uint32)
    if needs_transformation_data:
        V.mesh.topology.create_entity_permutations()
        cell_perms = V.mesh.topology.get_cell_permutation_info()
    if e0.needs_dof_transformations:
        raise NotImplementedError("Dof transformations not implemented")
    # Assemble over cells
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)
    nptype = "complex128" if is_complex else "float64"
    if num_cell_integrals > 0:
        V.mesh.topology.create_entity_permutations()
        for i, id in enumerate(subdomain_ids):
            cell_kernel = getattr(ufc_form.integrals(dolfinx.fem.IntegralType.cell)[i], f"tabulate_tensor_{nptype}")
            active_cells = cpp_form.domains(dolfinx.fem.IntegralType.cell, id)
            coeffs_i = form_coeffs[(dolfinx.fem.IntegralType.cell, id)]
            with vector.localForm() as b:
                assemble_cells(numpy.asarray(b), cell_kernel, active_cells[numpy.isin(active_cells, slave_cells)],
                               (pos, x_dofs, x), coeffs_i, form_consts,
                               cell_perms, dofs, block_size, num_dofs_per_element, mpc_data)

    # Assemble exterior facet integrals
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)
    if num_exterior_integrals > 0:
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        # Get facet permutations if required
        facet_perms = numpy.array([], dtype=numpy.uint8)
        if cpp_form.needs_facet_permutations:
            facet_perms = V.mesh.topology.get_facet_permutations()
        perm = (cell_perms, cpp_form.needs_facet_permutations, facet_perms)
        for i, id in enumerate(subdomain_ids):
            facet_kernel = getattr(ufc_form.integrals(dolfinx.fem.IntegralType.exterior_facet)[i],
                                   f"tabulate_tensor_{nptype}")
            coeffs_i = form_coeffs[(dolfinx.fem.IntegralType.exterior_facet, id)]
            facets = cpp_form.domains(dolfinx.fem.IntegralType.exterior_facet, id)
            facet_info = pack_slave_facet_info(facets, slave_cells)
            num_facets_per_cell = len(V.mesh.topology.connectivity(tdim, tdim - 1).links(0))
            with vector.localForm() as b:
                assemble_exterior_slave_facets(numpy.asarray(b), facet_kernel, facet_info, (pos, x_dofs, x),
                                               coeffs_i, form_consts, perm,
                                               dofs, block_size, num_dofs_per_element, mpc_data, num_facets_per_cell)
    timer_vector.stop()
    return vector


@numba.njit
def assemble_cells(b: 'numpy.ndarray[PETSc.ScalarType]',
                   kernel: ffi.CData, active_cells: 'numpy.ndarray[numpy.int32]',
                   mesh: Tuple['numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]',
                               'numpy.ndarray[numpy.float64]'],
                   coeffs: 'numpy.ndarray[PETSc.ScalarType]',
                   constants: 'numpy.ndarray[PETSc.ScalarType]',
                   permutation_info: 'numpy.ndarray[numpy.uint32]',
                   dofmap: 'numpy.ndarray[numpy.int32]',
                   block_size: int,
                   num_dofs_per_element: int,
                   mpc: Tuple['numpy.ndarray[numpy.int32]', 'numpy.ndarray[PETSc.ScalarType]',
                              'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]',
                              'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]']):
    """Assemble additional MPC contributions for cell integrals"""
    ffi_fb = ffi.from_buffer

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1] - pos[0], 3))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=PETSc.ScalarType)

    for cell_index in active_cells:
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        cell = pos[cell_index]

        # Compute mesh geometry for cell
        geometry[:, :] = x[x_dofmap[cell:cell + num_vertices]]

        # Assemble local element vector
        b_local.fill(0.0)
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry), ffi_fb(facet_index),
               ffi_fb(facet_perm))
        # FIXME: Here we need to add the apply_dof_transformation function

        # Modify global vector and local cell contributions
        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, b_local, b_local_copy, mpc, dofmap,
                                 block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[num_dofs_per_element * cell_index + j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit
def assemble_exterior_slave_facets(b: 'numpy.ndarray[PETSc.ScalarType]',
                                   kernel: ffi.CData,
                                   facet_info: 'numpy.ndarray[numpy.int32]',
                                   mesh: Tuple['numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]',
                                               'numpy.ndarray[numpy.float64]'],
                                   coeffs: 'numpy.ndarray[PETSc.ScalarType]',
                                   constants: 'numpy.ndarray[PETSc.ScalarType]',
                                   permutation_info: 'numpy.ndarray[numpy.uint32]',
                                   dofmap: 'numpy.ndarray[numpy.int32]',
                                   block_size: int,
                                   num_dofs_per_element: int,
                                   mpc: Tuple['numpy.ndarray[numpy.int32]', 'numpy.ndarray[PETSc.ScalarType]',
                                              'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]',
                                              'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]'],
                                   num_facets_per_cell: int):
    """Assemble additional MPC contributions for facets"""
    ffi_fb = ffi.from_buffer

    # Unpack facet permutation info
    cell_perms, needs_facet_perm, facet_perms = permutation_info
    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # Unpack mesh data
    pos, x_dofmap, x = mesh

    geometry = numpy.zeros((pos[1] - pos[0], 3))
    b_local = numpy.zeros(block_size * num_dofs_per_element, dtype=PETSc.ScalarType)
    for i in range(facet_info.shape[0]):
        # Extract cell index (local to process) and facet index (local to cell) for kernel
        cell_index, local_facet = facet_info[i]
        facet_index[0] = local_facet

        # Extract cell geometry
        cell = pos[cell_index]
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        geometry[:, :] = x[x_dofmap[cell:cell + num_vertices]]

        # Compute local facet kernel
        b_local.fill(0.0)
        if needs_facet_perm:
            facet_perm[0] = facet_perms[cell_index * num_facets_per_cell + local_facet]
        kernel(ffi_fb(b_local), ffi_fb(coeffs[cell_index, :]), ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm))
        # FIXME: Here we need to add the apply_dof_transformation

        # Modify local contributions and add global MPC contributions
        b_local_copy = b_local.copy()
        modify_mpc_contributions(b, cell_index, b_local, b_local_copy,
                                 mpc, dofmap, block_size, num_dofs_per_element)
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                position = dofmap[num_dofs_per_element * cell_index + j] * block_size + k
                b[position] += (b_local[j * block_size + k] - b_local_copy[j * block_size + k])


@numba.njit(cache=True)
def modify_mpc_contributions(b: 'numpy.ndarray[PETSc.ScalarType]', cell_index: int,
                             b_local: 'numpy.ndarray[PETSc.ScalarType]',
                             b_copy: 'numpy.ndarray[PETSc.ScalarType]',
                             mpc: Tuple['numpy.ndarray[numpy.int32]', 'numpy.ndarray[PETSc.ScalarType]',
                                        'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]',
                                        'numpy.ndarray[numpy.int32]', 'numpy.ndarray[numpy.int32]'],
                             dofmap: 'numpy.ndarray[numpy.int32]',
                             block_size: int,
                             num_dofs_per_element: int):
    """
    Modify local entries of b_local with MPC info and add modified
    entries to global vector b.
    """

    # Unwrap MPC data
    masters, coefficients, offsets, cell_to_slave, cell_to_slave_offset, is_slave = mpc

    # Determine which slaves are in this cell,
    # and which global index they have in 1D arrays
    cell_slaves = cell_to_slave[cell_to_slave_offset[cell_index]:
                                cell_to_slave_offset[cell_index + 1]]

    # Get local index of slaves in cell
    cell_blocks = dofmap[num_dofs_per_element * cell_index:
                         num_dofs_per_element * cell_index + num_dofs_per_element]
    local_index = numpy.empty(len(cell_slaves), dtype=numpy.int32)
    for i in range(num_dofs_per_element):
        for j in range(block_size):
            dof = cell_blocks[i] * block_size + j
            if is_slave[dof]:
                location = numpy.flatnonzero(cell_slaves == dof)[0]
                local_index[location] = i * block_size + j

    # Move contribution from each slave to the corresponding master dof
    # and zero out local b
    for local, slave in zip(local_index, cell_slaves):
        cell_masters = masters[offsets[slave]: offsets[slave + 1]]
        cell_coeffs = coefficients[offsets[slave]: offsets[slave + 1]]
        for m0, c0 in zip(cell_masters, cell_coeffs):
            b[m0] += c0 * b_copy[local]
            b_local[local] = 0
