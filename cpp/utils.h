// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MultiPointConstraint.h"
#include <Eigen/Dense>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx_mpc
{

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<PetscScalar>& a);

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_basis_functions(
    std::shared_ptr<const dolfinx::function::FunctionSpace> V,
    const Eigen::Ref<const Eigen::Array<double, 1, 3, Eigen::RowMajor>>& x,
    const int index);

/// Given a function space, compute its shared entities
std::map<std::int32_t, std::set<int>>
compute_shared_indices(std::shared_ptr<dolfinx::function::FunctionSpace> V);

/// Append diagonal entries to sparsity pattern
/// @param[in] pattern The sparsity pattern
/// @param[in] dofs The dofs that require diagonal additions
/// @param[in] block size of problem
void add_pattern_diagonal(dolfinx::la::SparsityPattern& pattern,
                          Eigen::Array<std::int32_t, Eigen::Dynamic, 1> blocks,
                          std::int32_t block_size);

dolfinx::la::PETScMatrix
create_matrix(const dolfinx::fem::Form<PetscScalar>& a,
              const std::shared_ptr<dolfinx_mpc::MultiPointConstraint> mpc);
/// Create neighborhood communicators from every processor with a slave dof on
/// it, to the processors with a set of master facets.
/// @param[in] meshtags The meshtag
/// @param[in] has_slaves Boolean saying if the processor owns slave dofs
/// @param[in] master_marker Tag for the other interface
std::array<MPI_Comm, 2>
create_neighborhood_comms(dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                          const bool has_slave, std::int32_t& master_marker);

/// Create neighbourhood communicators from local_dofs to processors who has
/// this as a ghost.
/// @param[in] local_dofs Vector of local dofs
/// @param[in] local_dofs Vector of ghost dofs
/// @param[in] index_map The index map relating procs and ghosts
MPI_Comm create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_dofs,
    std::vector<std::int32_t>& ghost_dofs,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map);

} // namespace dolfinx_mpc
