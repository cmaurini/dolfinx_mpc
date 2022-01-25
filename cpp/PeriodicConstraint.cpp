// Copyright (C) 2022 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#include "PeriodicConstraint.h"
#include "utils.h"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace
{

/// Given a mesh and corresponding bounding box tree and a set of points,check
/// which cells (local to process) collide with each point.
/// Return an array of the same size as the number of points, where the ith
/// entry corresponds to the first cell colliding with the ith point.
/// @note If no colliding point is found, the index -1 is returned.
/// @param[in] mesh The mesh
/// @param[in] tree The boundingbox tree of all cells (local to process) in the
/// mesh
/// @param[in] points The points to check collision with, shape (num_points, 3)
std::vector<std::int32_t>
find_local_collisions(const std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                      const dolfinx::geometry::BoundingBoxTree& tree,
                      const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);

  // Compute collisions for each point with BoundingBoxTree
  auto bbox_collisions = dolfinx::geometry::compute_collisions(tree, points);

  // Compute exact collision
  auto cell_collisions = dolfinx::geometry::compute_colliding_cells(
      *mesh.get(), bbox_collisions, points);

  // Extract first collision
  std::vector<std::int32_t> collisions(points.shape(0), -1);
  for (int i = 0; i < cell_collisions.num_nodes(); i++)
  {
    auto local_cells = cell_collisions.links(i);
    if (!local_cells.empty())
      collisions[i] = local_cells[0];
  }
  return collisions;
}

template <typename T>
std::vector<std::int32_t> remove_bc_blocks(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    tcb::span<const std::int32_t> blocks,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs)
{
  auto dofmap = V->dofmap();
  auto imap = dofmap->index_map;
  const int bs = dofmap->index_map_bs();
  std::int32_t dim = bs * (imap->size_local() + imap->num_ghosts());
  std::vector<std::int8_t> dof_marker(dim, false);

  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (V->contains(*bcs[k]->function_space()))
    {

      bcs[k]->mark_dofs(dof_marker);
    }
  }

  // Remove slave blocks contained in DirichletBC
  std::vector<std::int32_t> reduced_blocks;
  reduced_blocks.reserve(blocks.size());
  bool is_bc;
  for (auto block : blocks)
  {
    is_bc = false;
    for (int j = 0; j < bs; j++)
    {
      if (dof_marker[block * bs + j])
      {
        is_bc = true;
        break;
      }
    }
    if (!is_bc)
      reduced_blocks.push_back(block);
  }
  return reduced_blocks;
}

xt::xtensor<double, 2>
tabulate_dof_coordinates(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                         tcb::span<std::int32_t> dofs,
                         tcb::span<std::int32_t> cells)
{
  if (!V->component().empty())
  {
    throw std::runtime_error("Cannot tabulate coordinates for a "
                             "FunctionSpace that is a subspace.");
  }
  auto element = V->element();
  assert(element);
  if (V->element()->is_mixed())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a mixed FunctionSpace.");
  }

  auto mesh = V->mesh();
  assert(mesh);

  const std::size_t gdim = mesh->geometry().dim();

  // Get dofmap local size
  auto dofmap = V->dofmap();
  assert(dofmap);
  std::shared_ptr<const dolfinx::common::IndexMap> index_map
      = V->dofmap()->index_map;
  assert(index_map);

  const int element_block_size = element->block_size();
  const std::size_t scalar_dofs
      = element->space_dimension() / element_block_size;

  // Get the dof coordinates on the reference element
  if (!element->interpolation_ident())
  {
    throw std::runtime_error("Cannot evaluate dof coordinates - this element "
                             "does not have pointwise evaluation.");
  }
  const xt::xtensor<double, 2>& X = element->interpolation_points();

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  xtl::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = x_dofmap.num_links(0);

  // Array to hold coordinates to return
  const std::size_t shape_c0 = 3;
  const std::size_t shape_c1 = dofs.size();
  xt::xtensor<double, 2> coords = xt::zeros<double>({shape_c0, shape_c1});

  // Loop over cells and tabulate dofs
  xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  const std::function<void(const xtl::span<double>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    // Extract cell geometry
    auto x_dofs = x_dofmap.links(cells[c]);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate dof coordinates on cell
    dolfinx::fem::CoordinateElement::push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(xtl::span(x.data(), x.size()),
                             xtl::span(cell_info.data(), cell_info.size()),
                             (std::int32_t)c, (int)x.shape(1));

    // Get cell dofmap
    auto cell_dofs = dofmap->cell_dofs(cells[c]);
    auto it = std::find(cell_dofs.begin(), cell_dofs.end(), dofs[c]);
    auto loc = std::distance(cell_dofs.begin(), it);

    // Copy dof coordinates into vector
    for (std::size_t j = 0; j < gdim; ++j)
      coords(j, c) = x(loc, j);
  }

  return coords;
}

template <typename T>
dolfinx_mpc::mpc_data _create_periodic_condition(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    tcb::span<std::int32_t> slave_blocks,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    double scale)
{

  if (const std::size_t value_size
      = V->element()->value_size() / V->element()->block_size();
      value_size > 1)
    throw std::runtime_error(
        "Periodic conditions for vector valued spaces are not implemented");

  // Tolerance for adding scaled basis values to MPC. Any scaled basis
  // value with lower absolute value than the tolerance is ignored
  const double tol = 1e-13;

  auto mesh = V->mesh();
  auto dofmap = V->dofmap();
  auto imap = dofmap->index_map;
  const int bs = dofmap->index_map_bs();
  const int size_local = imap->size_local();
  std::vector<std::int32_t> ghost_owners = imap->ghost_owner_rank();

  // Get info about cells owned by the process
  const int tdim = mesh->topology().dim();
  auto cell_imap = mesh->topology().index_map(tdim);
  const int num_cells_local = cell_imap->size_local();
  const int num_ghost_cells = cell_imap->num_ghosts();

  // Only work with local blocks
  std::vector<std::int32_t> local_blocks;
  local_blocks.reserve(slave_blocks.size());
  std::for_each(slave_blocks.begin(), slave_blocks.end(),
                [&local_blocks, size_local](auto block)
                {
                  if (block < size_local)
                    local_blocks.push_back(block);
                });

  std::vector<std::int32_t> slave_cells;
  slave_cells.reserve(local_blocks.size());
  {
    // Create block -> cells map

    // Compute number of cells each dof is in
    std::vector<std::int32_t> num_cells_per_dof(size_local
                                                + ghost_owners.size());
    for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
    {
      auto dofs = dofmap->cell_dofs(i);
      for (auto dof : dofs)
        num_cells_per_dof[dof]++;
    }
    std::vector<std::int32_t> cell_dofs_disp(num_cells_per_dof.size() + 1, 0);
    std::partial_sum(num_cells_per_dof.begin(), num_cells_per_dof.end(),
                     cell_dofs_disp.begin() + 1);
    std::vector<std::int32_t> cell_map(cell_dofs_disp.back());
    // Reuse num_cells_per_dof for insertion
    std::fill(num_cells_per_dof.begin(), num_cells_per_dof.end(), 0);

    // Create the block -> cells map
    for (std::int32_t i = 0; i < num_cells_local + num_ghost_cells; i++)
    {
      auto dofs = dofmap->cell_dofs(i);
      for (auto dof : dofs)
        cell_map[cell_dofs_disp[dof] + num_cells_per_dof[dof]++] = i;
    }

    // Populate map from slaves to corresponding cell (choose first cell in map)
    std::for_each(local_blocks.begin(), local_blocks.end(),
                  [&cell_dofs_disp, &cell_map, &slave_cells](const auto dof)
                  { slave_cells.push_back(cell_map[cell_dofs_disp[dof]]); });
  }
  assert(slave_cells.size() == local_blocks.size());

  // Compute relation(slave_blocks)
  xt::xtensor<double, 2> mapped_T({local_blocks.size(), 3});
  {
    // Tabulate dof coordinates for each dof
    xt::xtensor<double, 2> x
        = tabulate_dof_coordinates(V, local_blocks, slave_cells);
    // Map all slave coordinates using the relation
    auto mapped_x = relation(x);
    mapped_T = xt::transpose(mapped_x);
  }

  // Create bounding-box tree over owned cells
  std::vector<std::int32_t> r(num_cells_local);
  std::iota(r.begin(), r.end(), 0);
  dolfinx::geometry::BoundingBoxTree tree(*mesh.get(), tdim, r, 1e-15);
  auto process_tree = tree.create_global_tree(mesh->comm());
  auto colliding_bbox_processes
      = dolfinx::geometry::compute_collisions(process_tree, mapped_T);

  std::vector<std::int32_t> local_cell_collisions
      = find_local_collisions(mesh, tree, mapped_T);
  dolfinx::common::Timer t0("~~Periodic: Local cell and eval basis");
  xt::xtensor<double, 3> tabulated_basis_values
      = dolfinx_mpc::evaluate_basis_functions(V, mapped_T,
                                              local_cell_collisions);
  t0.stop();
  // Create output arrays
  std::vector<std::int32_t> slaves;
  slaves.reserve(slave_blocks.size() * bs);
  std::vector<std::int64_t> masters;
  masters.reserve(slave_blocks.size() * bs);
  std::vector<std::int32_t> owners;
  owners.reserve(slave_blocks.size() * bs);
  // FIXME: This should really be templated
  // Requires changes all over the place for mpc_data
  std::vector<PetscScalar> coeffs;
  coeffs.reserve(slave_blocks.size() * bs);
  std::vector<std::int32_t> num_masters_per_slave;
  num_masters_per_slave.reserve(slave_blocks.size() * bs);

  // Temporary array holding global indices
  std::vector<std::int64_t> global_blocks;

  const int rank = dolfinx::MPI::rank(mesh->comm());
  // Create counter for blocks that will be sent to other processes
  const int num_procs = dolfinx::MPI::size(mesh->comm());
  std::vector<std::int32_t> off_process_counter(num_procs, 0);
  for (std::size_t i = 0; i < local_cell_collisions.size(); i++)
  {
    if (local_cell_collisions[i] != -1)
    {
      // Compute basis functions at mapped point
      const std::int32_t cell = local_cell_collisions[i];
      auto basis_values
          = xt::view(tabulated_basis_values, i, xt::all(), xt::all());

      // Map local dofs on master cell to global indices
      auto cell_blocks = dofmap->cell_dofs(cell);
      global_blocks.resize(cell_blocks.size());
      imap->local_to_global(cell_blocks, global_blocks);

      // Check if basis values are not zero, and add master, coeff and owner
      // info for each dof in the block
      for (int b = 0; b < bs; b++)
      {
        slaves.push_back(local_blocks[i] * bs + b);
        int num_masters = 0;
        for (std::size_t j = 0; j < cell_blocks.size(); j++)
        {
          const std::int32_t cell_block = cell_blocks[j];
          // NOTE: Assuming 0 value size
          if (const double val = scale * basis_values(j, 0);
              std::abs(val) > tol)
          {
            num_masters++;
            masters.push_back(global_blocks[j] * bs + b);
            coeffs.push_back(val);
            if (cell_block < size_local)
              owners.push_back(rank);
            else
              owners.push_back(ghost_owners[cell_block - size_local]);
          }
        }
        num_masters_per_slave.push_back(num_masters);
      }
    }
    else
    {
      // Count number of incoming blocks from other processes
      auto procs = colliding_bbox_processes.links((int)i);
      for (auto proc : procs)
      {
        if (rank == proc)
          continue;
        off_process_counter[proc]++;
      }
    }
  }
  // Communicate s_to_m
  std::vector<std::int32_t> s_to_m_ranks;
  std::vector<std::int8_t> s_to_m_indicator(num_procs, 0);
  std::vector<std::int32_t> num_out_slaves;
  for (int i = 0; i < num_procs; i++)
  {
    if (off_process_counter[i] > 0 && i != rank)
    {
      s_to_m_indicator[i] = 1;
      s_to_m_ranks.push_back(i);
      num_out_slaves.push_back(off_process_counter[i]);
    }
  }

  std::vector<std::int8_t> indicator(num_procs);
  MPI_Alltoall(s_to_m_indicator.data(), 1, MPI_INT8_T, indicator.data(), 1,
               MPI_INT8_T, mesh->comm());

  std::vector<std::int32_t> m_to_s_ranks;
  for (int i = 0; i < num_procs; i++)
    if (indicator[i])
      m_to_s_ranks.push_back(i);

  // Create neighborhood communicator
  // Slave block owners -> Process with possible masters
  std::vector<int> s_to_m_weights(s_to_m_ranks.size(), 1);
  std::vector<int> m_to_s_weights(m_to_s_ranks.size(), 1);
  auto slave_to_master = MPI_COMM_NULL;
  MPI_Dist_graph_create_adjacent(
      mesh->comm(), (int)m_to_s_ranks.size(), m_to_s_ranks.data(),
      m_to_s_weights.data(), (int)s_to_m_ranks.size(), s_to_m_ranks.data(),
      s_to_m_weights.data(), MPI_INFO_NULL, false, &slave_to_master);

  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(slave_to_master, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == (int)m_to_s_ranks.size());
  assert(outdegree == (int)s_to_m_ranks.size());

  // Compute number of receiving slaves
  std::vector<std::int32_t> num_recv_slaves(indegree);
  MPI_Neighbor_alltoall(
      num_out_slaves.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_recv_slaves.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      slave_to_master);

  // Prepare data structures for sending information
  std::vector<int> disp_out(outdegree + 1, 0);
  std::partial_sum(num_out_slaves.begin(), num_out_slaves.end(),
                   disp_out.begin() + 1);

  // Prepare data structures for receiving information
  std::vector<int> disp_in(indegree + 1, 0);
  std::partial_sum(num_recv_slaves.begin(), num_recv_slaves.end(),
                   disp_in.begin() + 1);

  // Array holding all dofs (index local to process) for coordinates sent
  // out
  std::vector<std::int32_t> searching_dofs(bs * disp_out.back());

  // Communicate coordinates of slave blocks
  std::vector<std::int32_t> out_placement(outdegree, 0);
  xt::xtensor<double, 2> coords_out({(std::size_t)disp_out.back(), 3});

  for (std::size_t i = 0; i < local_cell_collisions.size(); i++)
  {
    if (local_cell_collisions[i] == -1)
    {
      auto procs = colliding_bbox_processes.links((int)i);
      for (auto proc : procs)
      {
        if (rank == proc)
          continue;
        // Find position in neighborhood communicator
        auto it = std::find(s_to_m_ranks.begin(), s_to_m_ranks.end(), proc);
        assert(it != s_to_m_ranks.end());
        auto dist = std::distance(s_to_m_ranks.begin(), it);
        const std::int32_t insert_location
            = disp_out[dist] + out_placement[dist]++;
        xt::row(coords_out, insert_location) = xt::row(mapped_T, i);
        for (int b = 0; b < bs; b++)
          searching_dofs[insert_location * bs + b] = local_blocks[i] * bs + b;
      }
    }
  }

  // Communciate coordinates with other process
  xt::xtensor<double, 2> coords_recv({(std::size_t)disp_in.back(), 3});

  // Take into account that we send three values per slave
  auto m_3 = [](auto& num) { num *= 3; };
  std::for_each(disp_out.begin(), disp_out.end(), m_3);
  std::for_each(num_out_slaves.begin(), num_out_slaves.end(), m_3);
  std::for_each(disp_in.begin(), disp_in.end(), m_3);
  std::for_each(num_recv_slaves.begin(), num_recv_slaves.end(), m_3);

  // Communicate coordinates
  MPI_Neighbor_alltoallv(coords_out.data(), num_out_slaves.data(),
                         disp_out.data(), dolfinx::MPI::mpi_type<double>(),
                         coords_recv.data(), num_recv_slaves.data(),
                         disp_in.data(), dolfinx::MPI::mpi_type<double>(),
                         slave_to_master);

  // Reset in_displacements to be per block for later usage
  auto d_3 = [](auto& num) { num /= 3; };
  std::for_each(disp_in.begin(), disp_in.end(), d_3);

  // Reset out_displacments to be for every slave
  auto m_bs_d_3 = [bs](auto& num) { num = num * bs / 3; };
  std::for_each(disp_out.begin(), disp_out.end(), m_bs_d_3);
  std::for_each(num_out_slaves.begin(), num_out_slaves.end(), m_bs_d_3);

  // Create remote arrays
  std::vector<std::int64_t> masters_remote;
  masters_remote.reserve(coords_recv.size());
  std::vector<std::int32_t> owners_remote;
  owners_remote.reserve(coords_recv.size());
  std::vector<PetscScalar> coeffs_remote;
  coeffs_remote.reserve(coords_recv.size());
  std::vector<std::int32_t> num_masters_per_slave_remote;
  num_masters_per_slave_remote.reserve(bs * coords_recv.size() / 3);

  std::vector<std::int32_t> remote_cell_collisions
      = find_local_collisions(mesh, tree, coords_recv);
  xt::xtensor<double, 3> remote_basis_values
      = dolfinx_mpc::evaluate_basis_functions(V, coords_recv,
                                              remote_cell_collisions);

  // Find remote masters and count how many to send to each process
  std::vector<std::int32_t> num_remote_masters(indegree, 0);
  std::vector<std::int32_t> num_remote_slaves(indegree);
  for (int i = 0; i < indegree; i++)
  {
    // Count number of masters added and number of slaves for output
    std::int32_t r_masters = 0;
    num_remote_slaves[i] = bs * (disp_in[i + 1] - disp_in[i]);

    for (std::int32_t j = disp_in[i]; j < disp_in[i + 1]; j++)
    {
      if (const std::int32_t cell = remote_cell_collisions[j]; j == -1)
      {
        for (int b = 0; b < bs; b++)
          num_masters_per_slave_remote.push_back(0);
      }
      else
      {
        // Compute basis functions at mapped point
        // Check if basis values are not zero, and add master, coeff and
        // owner info for each dof in the block
        auto cell_blocks = dofmap->cell_dofs(cell);
        global_blocks.resize(cell_blocks.size());
        imap->local_to_global(cell_blocks, global_blocks);
        auto remote_basis
            = xt::view(remote_basis_values, j, xt::all(), xt::all());

        for (int b = 0; b < bs; b++)
        {
          int num_masters = 0;
          for (std::size_t k = 0; k < cell_blocks.size(); k++)
          {
            // NOTE: Assuming value_size 0
            if (const double val = scale * remote_basis(k, 0);
                std::abs(val) > tol)
            {
              num_masters++;
              masters_remote.push_back(global_blocks[k] * bs + b);
              coeffs_remote.push_back(val);
              if (cell_blocks[k] < size_local)
                owners_remote.push_back(rank);
              else
                owners_remote.push_back(
                    ghost_owners[cell_blocks[k] - size_local]);
            }
          }
          r_masters += num_masters;
          num_masters_per_slave_remote.push_back(num_masters);
        }
      }
    }
    num_remote_masters[i] += r_masters;
  }

  // Create inverse communicator
  // Procs with possible masters -> slave block owners
  auto master_to_slave = MPI_COMM_NULL;
  MPI_Dist_graph_create_adjacent(
      mesh->comm(), (int)s_to_m_ranks.size(), s_to_m_ranks.data(),
      s_to_m_weights.data(), (int)m_to_s_ranks.size(), m_to_s_ranks.data(),
      m_to_s_weights.data(), MPI_INFO_NULL, false, &master_to_slave);

  // Send data back to owning process
  dolfinx_mpc::recv_data recv_data
      = dolfinx_mpc::send_master_data_to_owner<PetscScalar>(
          master_to_slave, num_remote_masters, num_remote_slaves,
          num_out_slaves, num_masters_per_slave_remote, masters_remote,
          coeffs_remote, owners_remote);

  // Append found slaves/master pairs
  dolfinx_mpc::append_master_data<PetscScalar>(
      recv_data, searching_dofs, slaves, masters, coeffs, owners,
      num_masters_per_slave, size_local, bs);

  // Distribute ghost data
  dolfinx_mpc::mpc_data ghost_data
      = dolfinx_mpc::distribute_ghost_data<PetscScalar>(
          slaves, masters, coeffs, owners, num_masters_per_slave, imap, bs);

  // Add ghost data to existing arrays
  std::vector<std::int32_t>& ghost_slaves = ghost_data.slaves;
  slaves.insert(std::end(slaves), std::begin(ghost_slaves),
                std::end(ghost_slaves));
  std::vector<std::int64_t>& ghost_masters = ghost_data.masters;
  masters.insert(std::end(masters), std::begin(ghost_masters),
                 std::end(ghost_masters));
  std::vector<std::int32_t>& ghost_num = ghost_data.offsets;
  num_masters_per_slave.insert(std::end(num_masters_per_slave),
                               std::begin(ghost_num), std::end(ghost_num));
  std::vector<PetscScalar>& ghost_coeffs = ghost_data.coeffs;
  coeffs.insert(std::end(coeffs), std::begin(ghost_coeffs),
                std::end(ghost_coeffs));
  std::vector<std::int32_t>& ghost_owner_ranks = ghost_data.owners;
  owners.insert(std::end(owners), std::begin(ghost_owner_ranks),
                std::end(ghost_owner_ranks));

  // Compute offsets
  std::vector<std::int32_t> offsets(num_masters_per_slave.size() + 1, 0);
  std::partial_sum(num_masters_per_slave.begin(), num_masters_per_slave.end(),
                   offsets.begin() + 1);

  dolfinx_mpc::mpc_data output;
  output.offsets = offsets;
  output.masters = masters;
  output.coeffs = coeffs;
  output.owners = owners;
  output.slaves = slaves;
  return output;
}

} // namespace

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        indicator,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale)
{
  std::vector<std::int32_t> slave_blocks
      = dolfinx::fem::locate_dofs_geometrical(*V.get(), indicator);

  // Remove blocks in Dirichlet bcs
  std::vector<std::int32_t> reduced_blocks
      = remove_bc_blocks<double>(V, slave_blocks, bcs);
  return _create_periodic_condition<double>(V, tcb::make_span(reduced_blocks),
                                            relation, scale);
};

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_geometrical(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::function<xt::xtensor<bool, 1>(const xt::xtensor<double, 2>&)>&
        indicator,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale)
{
  std::vector<std::int32_t> slave_blocks
      = dolfinx::fem::locate_dofs_geometrical(*V.get(), indicator);

  // Remove blocks in Dirichlet bcs
  std::vector<std::int32_t> reduced_blocks
      = remove_bc_blocks<std::complex<double>>(V, slave_blocks, bcs);
  return _create_periodic_condition<std::complex<double>>(
      V, tcb::make_span(reduced_blocks), relation, scale);
};

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<double>>>&
        bcs,
    double scale)
{

  std::vector<std::int32_t> entities = meshtag->find(tag);
  std::vector<std::int32_t> slave_blocks
      = dolfinx::fem::locate_dofs_topological(*V.get(), meshtag->dim(),
                                              entities);
  // Remove blocks in Dirichlet bcs
  std::vector<std::int32_t> reduced_blocks
      = remove_bc_blocks<double>(V, slave_blocks, bcs);
  return _create_periodic_condition<double>(V, tcb::make_span(reduced_blocks),
                                            relation, scale);
};

dolfinx_mpc::mpc_data dolfinx_mpc::create_periodic_condition_topological(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>> meshtag,
    const std::int32_t tag,
    const std::function<xt::xarray<double>(const xt::xtensor<double, 2>&)>&
        relation,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<std::complex<double>>>>&
        bcs,
    double scale)
{

  std::vector<std::int32_t> entities = meshtag->find(tag);
  std::vector<std::int32_t> slave_blocks
      = dolfinx::fem::locate_dofs_topological(*V.get(), meshtag->dim(),
                                              entities);

  // Remove blocks in Dirichlet bcs
  std::vector<std::int32_t> reduced_blocks
      = remove_bc_blocks<std::complex<double>>(V, slave_blocks, bcs);

  return _create_periodic_condition<std::complex<double>>(
      V, tcb::make_span(reduced_blocks), relation, scale);
};
