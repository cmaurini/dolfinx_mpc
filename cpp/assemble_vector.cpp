// Copyright (C) 2021 Jorgen S. Dokken & Nathan Sime
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector.h"
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <iostream>

namespace
{

/// Given a local element vector, move all slave contributions to the global
/// (local to process) vector.
/// @param [in, out] b The global (local to process) vector
/// @param [in, out] b_local The local element vector
/// @param [in] b_local_copy Copy of the local element vector
/// @param [in] num_dofs The number of degrees of freedom in the local vector
/// @param [in] bs The element block size
/// @param [in] slave_indices Indices of slaves in the local element vector
/// (relative to the multi point constraint)
/// @param[in] local_indices The slave indices (local to process)
/// @param[in] masters Adjacency list with master dofs
/// @param[in] coeffs Adjacency list with the master coefficients
template <typename T>
void modify_mpc_vec(
    const xtl::span<T>& b, const xtl::span<T>& b_local,
    const xtl::span<T>& b_local_org, const int num_dofs, const int bs,
    const xtl::span<const int32_t>& slave_indices,
    const std::vector<std::int32_t>& local_indices,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        masters,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>>& coeffs)
{

  // Given the set of slave indices in the cell, flatten all slaves, masters and
  // coefficients to equal length arrays for easy insertion
  std::vector<std::int32_t> flattened_masters;
  std::vector<std::int32_t> flattened_slaves;
  std::vector<std::int32_t> slaves_loc;
  std::vector<T> flattened_coeffs;
  for (std::int32_t i = 0; i < slave_indices.size(); ++i)
  {
    xtl::span<const std::int32_t> local_masters
        = masters->links(slave_indices[i]);
    xtl::span<const T> local_coeffs = coeffs->links(slave_indices[i]);

    for (std::int32_t j = 0; j < local_masters.size(); ++j)
    {
      slaves_loc.push_back(i);
      flattened_slaves.push_back(local_indices[i]);
      flattened_masters.push_back(local_masters[j]);
      flattened_coeffs.push_back(local_coeffs[j]);
    }
  }

  // Loop over all masters and move contributions that are outside the local
  // cell into b. We remove contributions in local in-out element vector
  for (std::int32_t slv_idx = 0; slv_idx < flattened_slaves.size(); ++slv_idx)
  {
    const std::int32_t& local_index = flattened_slaves[slv_idx];
    const std::int32_t& master = flattened_masters[slv_idx];
    const T& coeff = flattened_coeffs[slv_idx];

    for (std::int32_t i = 0; i < num_dofs; ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        b[master] += coeff * b_local_org[i * bs + j];
        b_local[i * bs + j] = 0;
      }
    }
  }
}

/// Assemble an integration kernel over a set of active entities, described
/// through into vector of type T, and apply the multipoint constraint
/// @param[in, out] b The vector to assemble into
/// @param[in] active_entities The set of active entities.
/// @param[in] dofmap The dofmap
/// @param[in] bs The block size of the dofmap
/// @param[in] coeffs The packed coefficients for all cells
/// @param[in] constants The pack constants
/// @param[in] cell_info The cell permutation info
/// @param[in] mpc The multipoint constraint
/// @param[in] fetch_cells Function that fetches the cell index for an entity
/// in active_entities
/// @param[in] assemble_local_element_matrix Function f(be, index) that
/// assembles into a local element matrix for a given entity
template <typename T, typename E_DESC>
void _assemble_entities_impl(
    xtl::span<T> b, const dolfinx::mesh::Mesh& mesh,
    const std::vector<E_DESC>& active_entities,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const xtl::span<const T> coeffs, int cstride,
    const std::vector<T>& constants,
    const xtl::span<const std::uint32_t>& cell_info,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc,
    const std::function<const std::int32_t(const E_DESC&)> fetch_cells,
    const std::function<void(xtl::span<T>, E_DESC)>
        assemble_local_element_vector)
{
  // Get MPC data
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      masters = mpc->masters_local();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<T>> coefficients
      = mpc->coeffs();
  const xtl::span<const std::int32_t> slaves = mpc->slaves();

  xtl::span<const std::int32_t> slave_cells = mpc->slave_cells();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      cell_to_slaves = mpc->cell_to_slaves();

  // Compute local indices for slave cells
  std::vector<bool> is_slave_entity
      = std::vector<bool>(active_entities.size(), false);
  std::vector<std::int32_t> slave_cell_index
      = std::vector<std::int32_t>(active_entities.size(), -1);
  for (std::int32_t i = 0; i < active_entities.size(); ++i)
  {
    const std::int32_t cell = fetch_cells(active_entities[i]);
    for (std::int32_t j = 0; j < slave_cells.size(); ++j)
      if (slave_cells[j] == cell)
      {
        is_slave_entity[i] = true;
        slave_cell_index[i] = j;
        break;
      }
  }

  // NOTE: Assertion that all links have the same size (no P refinement)
  const int num_dofs = dofmap.links(0).size();
  std::vector<T> be(bs * num_dofs);
  const xtl::span<T> _be(be);
  std::vector<T> be_copy(bs * num_dofs);
  const xtl::span<T> _be_copy(be_copy);

  // Local vector to indicate which dof is slave
  std::vector<bool> is_slave(bs * num_dofs);

  // Assemble over all entities
  for (std::int32_t e = 0; e < active_entities.size(); ++e)
  {
    // Assemble into element vector
    assemble_local_element_vector(_be, active_entities[e]);

    const std::int32_t cell = fetch_cells(active_entities[e]);
    auto dofs = dofmap.links(cell);

    // Modify local element matrix if entity is connected to a slave cell
    if (is_slave_entity[e])
    {
      // Find local position of every slave in cell
      xtl::span<const int32_t> slave_indices
          = cell_to_slaves->links(slave_cell_index[e]);
      std::vector<std::int32_t> local_indices(slave_indices.size());
      for (std::int32_t i = 0; i < slave_indices.size(); ++i)
      {
        bool found = false;
        for (std::int32_t j = 0; j < dofs.size(); ++j)
        {
          for (std::int32_t k = 0; k < bs; ++k)
          {
            if (bs * dofs[j] + k == slaves[slave_indices[i]])
            {
              local_indices[i] = bs * j + k;
              break;
            }
          }
          if (found)
            break;
        }
      }
      // Modify element vector for MPC and insert into b for non-local
      // contributions
      std::copy(be.begin(), be.end(), be_copy.begin());
      modify_mpc_vec<T>(b, _be, _be_copy, num_dofs, bs, slave_indices,
                        local_indices, masters, coefficients);
    }
    // Add local contribution to b
    for (int i = 0; i < num_dofs; ++i)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs[i] + k] += be[bs * i + k];
  }
}

template <typename T>
void _assemble_vector(
    xtl::span<T> b, const dolfinx::fem::Form<T>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<T>>& mpc)
{

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap
      = L.function_spaces().at(0)->dofmap();
  assert(dofmap);
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();

  const int num_dofs = dofs.links(0).size();
  const std::uint32_t ndim = bs * num_dofs;

  // Prepare constants & coefficients
  const std::vector<T> constants = pack_constants(L);
  const auto coeffs = dolfinx::fem::pack_coefficients(L);

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Prepare dof tranformation data
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = L.function_spaces().at(0)->element();
  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform = element->get_dof_transformation_function<T>();
  const bool needs_transformation_data
      = element->needs_dof_transformations() or L.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  if (L.num_integrals(dolfinx::fem::IntegralType::cell) > 0)
  {
    const auto fetch_cells = [&](const std::int32_t& entity) { return entity; };
    for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
    {
      const auto& fn = L.kernel(dolfinx::fem::IntegralType::cell, i);

      /// Assemble local cell kernels into a vector
      /// @param[in] be The local element vector
      /// @param[in] cell The cell index
      const auto assemble_local_cell_vector
          = [&](xtl::span<T> be, std::int32_t cell)
      {
        // Fetch the coordiantes of the cell
        const xtl::span<const std::int32_t> x_dofs = x_dofmap.links(cell);
        const xt::xarray<double> coordinate_dofs(
            xt::view(x_g, xt::keep(x_dofs), xt::all()));

        // Tabulate tensor
        std::fill(be.data(), be.data() + be.size(), 0);
        fn(be.data(), coeffs.first.data() + cell * coeffs.second,
           constants.data(), coordinate_dofs.data(), nullptr, nullptr);

        // Apply any required transformations
        dof_transform(be, cell_info, cell, 1);
      };

      // Assemble over all active cells
      const std::vector<std::int32_t>& active_cells = L.cell_domains(i);
      _assemble_entities_impl<T, std::int32_t>(
          b, *mesh, active_cells, dofs, bs, coeffs.first, coeffs.second,
          constants, cell_info, mpc, fetch_cells, assemble_local_cell_vector);
    }
  }
}
} // namespace
//-----------------------------------------------------------------------------

void dolfinx_mpc::assemble_vector(
    xtl::span<double> b, const dolfinx::fem::Form<double>& L,
    const std::shared_ptr<const dolfinx_mpc::MultiPointConstraint<double>>& mpc)
{
  _assemble_vector<double>(b, L, mpc);
}
//-----------------------------------------------------------------------------
