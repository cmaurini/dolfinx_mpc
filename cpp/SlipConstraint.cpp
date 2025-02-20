// Copyright (C) 2019-2021 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "SlipConstraint.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
using namespace dolfinx_mpc;

mpc_data dolfinx_mpc::create_slip_condition(
    std::shared_ptr<dolfinx::fem::FunctionSpace>& space,
    const dolfinx::mesh::MeshTags<std::int32_t>& meshtags, std::int32_t marker,
    const dolfinx::fem::Function<PetscScalar>& v,
    std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>
        bcs,
    const bool sub_space)
{
  // Map from collapsed sub space to parent space
  std::function<const std::int32_t(const std::int32_t&)> parent_map;

  // Interpolate input function into suitable space
  std::shared_ptr<dolfinx::fem::Function<PetscScalar>> n;
  if (sub_space)
  {
    std::pair<dolfinx::fem::FunctionSpace, std::vector<int32_t>> collapsed_space
        = space->collapse();
    auto V_ptr = std::make_shared<dolfinx::fem::FunctionSpace>(
        std::move(collapsed_space.first));
    n = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V_ptr);
    n->interpolate(v);
    parent_map = [sub_map = collapsed_space.second](const std::int32_t dof)
    { return sub_map[dof]; };
  }
  else
  {
    parent_map = [](const std::int32_t dof) { return dof; };
    n = std::make_shared<dolfinx::fem::Function<PetscScalar>>(space);
    n->interpolate(v);
  }

  // Create view of n as xtensor
  const tcb::span<const PetscScalar>& n_vec = n->x()->array();
  std::array<std::size_t, 1> shape = {n_vec.size()};
  auto n_xt = xt::adapt(n_vec.data(), n_vec.size(), xt::no_ownership(), shape);

  // Info from parent space
  auto W_imap = space->dofmap()->index_map;
  std::vector<std::int32_t> W_ghost_owners = W_imap->ghost_owner_rank();
  const int W_bs = space->dofmap()->index_map_bs();
  const int W_local_size = W_imap->size_local();

  const std::vector<std::int32_t> slave_facets = meshtags.find(marker);

  auto mesh = space->mesh();
  MPI_Comm comm = mesh->comm();
  const int rank = dolfinx::MPI::rank(comm);

  // Array containing blocks of the MPC slaves
  std::vector<std::int32_t> slave_blocks;
  std::int32_t num_normal_components;

  // Find blocks in collapsed space and remove DirichletBC dofs
  if (sub_space)
  {

    // Get all degrees of freedom in the sub-space on the given facets
    std::array<std::vector<std::int32_t>, 2> entity_dofs
        = dolfinx::fem::locate_dofs_topological({*space, *n->function_space()},
                                                meshtags.dim(), slave_facets);

    // Remove Dirichlet BC dofs
    const std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<PetscScalar>(*space, entity_dofs[0], bcs);
    num_normal_components = n->function_space()->dofmap()->index_map_bs();
    slave_blocks.reserve(entity_dofs[0].size());
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        slave_blocks.push_back(entity_dofs[1][i] / num_normal_components);
    // Erase blocks duplicate blocks
    dolfinx::radix_sort(xtl::span<std::int32_t>(slave_blocks));
    slave_blocks.erase(std::unique(slave_blocks.begin(), slave_blocks.end()),
                       slave_blocks.end());
  }
  else
  {
    num_normal_components = W_bs;
    std::vector<std::int32_t> all_slave_blocks
        = dolfinx::fem::locate_dofs_topological(*space, meshtags.dim(),
                                                slave_facets);
    // Remove Dirichlet BC dofs
    const std::vector<std::int8_t> bc_marker
        = dolfinx_mpc::is_bc<PetscScalar>(*space, all_slave_blocks, bcs);
    for (std::size_t i = 0; i < bc_marker.size(); i++)
      if (!bc_marker[i])
        slave_blocks.push_back(all_slave_blocks[i]);
  }
  // Create array to hold degrees of freedom from the subspace of v for each
  // block
  std::vector<std::int32_t> normal_dofs(num_normal_components);

  // Arrays holding MPC data
  std::vector<std::int32_t> slaves;
  std::vector<std::int64_t> masters;
  std::vector<PetscScalar> coeffs;
  std::vector<std::int32_t> owners;
  std::vector<std::int32_t> offsets(1, 0);

  // Temporary arrays used to hold information about masters
  std::vector<std::int64_t> pair_m;
  for (auto block : slave_blocks)
  {
    // Obtain degrees of freedom for normal vector at slave dof
    for (std::int32_t i = 0; i < num_normal_components; ++i)
      normal_dofs[i] = block * num_normal_components + i;
    // Determine slave dof by component with biggest normal vector (to avoid
    // issues with grids aligned with coordiante system)
    auto normal = xt::view(n_xt, xt::keep(normal_dofs));
    std::int32_t slave_index = xt::argmax(xt::abs(normal))(0, 0);

    std::int32_t parent_slave
        = parent_map(block * num_normal_components + slave_index);
    slaves.push_back(parent_slave);

    std::vector<std::int32_t> parent_masters;
    std::vector<PetscScalar> pair_c;
    std::vector<std::int32_t> pair_o;
    for (std::int32_t i = 0; i < num_normal_components; ++i)
    {
      if (i != slave_index)
      {
        const std::int32_t parent_dof
            = parent_map(block * num_normal_components + i);
        PetscScalar coeff = -normal[i] / normal[slave_index];
        parent_masters.push_back(parent_dof);
        pair_c.push_back(coeff);
        const std::int32_t m_rank
            = parent_dof < W_local_size * W_bs
                  ? rank
                  : W_ghost_owners[parent_dof / W_bs - W_local_size];
        pair_o.push_back(m_rank);
      }
    }
    // Convert local parent dof to local parent block
    std::vector<std::int32_t> parent_blocks = parent_masters;
    std::for_each(parent_blocks.begin(), parent_blocks.end(),
                  [W_bs](auto& b) { b /= W_bs; });
    // Map blocks from local to global
    pair_m.resize(parent_masters.size());
    W_imap->local_to_global(parent_blocks, pair_m);
    // Convert global parent block to the dofs
    for (std::size_t i = 0; i < parent_masters.size(); ++i)
    {
      std::div_t div = std::div(parent_masters[i], W_bs);
      pair_m[i] = pair_m[i] * W_bs + div.rem;
    }
    masters.insert(masters.end(), pair_m.begin(), pair_m.end());
    coeffs.insert(coeffs.end(), pair_c.begin(), pair_c.end());
    offsets.push_back(masters.size());
    owners.insert(owners.end(), pair_o.begin(), pair_o.end());
  }

  mpc_data data;
  data.slaves = slaves;

  data.masters = masters;
  data.offsets = offsets;
  data.owners = owners;
  data.coeffs = coeffs;
  return data;
}