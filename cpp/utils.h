// Copyright (C) 2020 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "MultiPointConstraint.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
namespace dolfinx_mpc
{

/// Structure to hold data after mpi communication
template <typename T>
struct recv_data
{
  std::vector<std::int32_t> num_masters_per_slave;
  std::vector<std::int64_t> masters;
  std::vector<std::int32_t> owners;
  std::vector<T> coeffs;
};

template <typename T>
class MultiPointConstraint;

/// Get basis values for all degrees at point x in a given cell
/// @param[in] V       The function space
/// @param[in] x       The physical coordinate
/// @param[in] index   The cell_index
xt::xtensor<double, 2>
get_basis_functions(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                    const xt::xtensor<double, 2>& x, const int index);

/// Given a function space, compute its shared entities
std::map<std::int32_t, std::set<int>>
compute_shared_indices(std::shared_ptr<dolfinx::fem::FunctionSpace> V);

dolfinx::la::petsc::Matrix create_matrix(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc,
    const std::string& type = std::string());
/// Create neighborhood communicators from every processor with a slave dof on
/// it, to the processors with a set of master facets.
/// @param[in] meshtags The meshtag
/// @param[in] has_slaves Boolean saying if the processor owns slave dofs
/// @param[in] master_marker Tag for the other interface
std::array<MPI_Comm, 2>
create_neighborhood_comms(dolfinx::mesh::MeshTags<std::int32_t>& meshtags,
                          const bool has_slave, std::int32_t& master_marker);

/// Create neighbourhood communicators from a set of local indices to process
/// who has these indices as ghosts.
/// @param[in] local_dofs Vector of local blocks
/// @param[in] ghost_dofs Vector of ghost blocks
/// @param[in] index_map The index map relating procs and ghosts
MPI_Comm create_owner_to_ghost_comm(
    std::vector<std::int32_t>& local_blocks,
    std::vector<std::int32_t>& ghost_blocks,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map);

/// Creates a normal approximation for the dofs in the closure of the attached
/// facets, where the normal is an average if a dof belongs to multiple facets
dolfinx::fem::Function<PetscScalar>
create_normal_approximation(std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                            std::int32_t dim,
                            const xtl::span<const std::int32_t>& entities);

/// Append standard sparsity pattern for a given form to a pre-initialized
/// pattern and a DofMap
/// @param[in] pattern The sparsity pattern
/// @param[in] a       The variational formulation
template <typename T>
void build_standard_pattern(dolfinx::la::SparsityPattern& pattern,
                            const dolfinx::fem::Form<T>& a)
{

  dolfinx::common::Timer timer("~MPC: Create sparsity pattern (Classic)");
  // Get dof maps
  std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2>
      dofmaps{*a.function_spaces().at(0)->dofmap(),
              *a.function_spaces().at(1)->dofmap()};

  // Get mesh
  assert(a.mesh());
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  if (a.integral_ids(dolfinx::fem::IntegralType::cell).size() > 0)
  {
    dolfinx::fem::sparsitybuild::cells(pattern, mesh.topology(),
                                       {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integral_ids(dolfinx::fem::IntegralType::interior_facet).size() > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::sparsitybuild::interior_facets(pattern, mesh.topology(),
                                                 {{dofmaps[0], dofmaps[1]}});
  }

  if (a.integral_ids(dolfinx::fem::IntegralType::exterior_facet).size() > 0)
  {
    mesh.topology_mutable().create_entities(mesh.topology().dim() - 1);
    mesh.topology_mutable().create_connectivity(mesh.topology().dim() - 1,
                                                mesh.topology().dim());
    dolfinx::fem::sparsitybuild::exterior_facets(pattern, mesh.topology(),
                                                 {{dofmaps[0], dofmaps[1]}});
  }
}

/// Add sparsity pattern for multi-point constraints to existing
/// sparsity pattern
/// @param[in] a bi-linear form for the current variational problem
/// (The one used to generate input sparsity-pattern)
/// @param[in] mpc The multi point constraint.
dolfinx::la::SparsityPattern create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a,
    const std::shared_ptr<dolfinx_mpc::MultiPointConstraint<PetscScalar>> mpc);

/// Compute the dot product u . vs
/// @param u The first vector. It must has size 3.
/// @param v The second vector. It must has size 3.
/// @return The dot product `u . v`. The type will be the same as value size
/// of u.
template <typename U, typename V>
typename U::value_type dot(const U& u, const V& v)
{
  assert(u.size() == 3);
  assert(v.size() == 3);
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
};

/// Given a list of global degrees of freedom, map them to their local index
/// @param[in] V The original function space
/// @param[in] global_dofs The list of dofs (global index)
/// @returns List of local dofs
std::vector<std::int32_t>
map_dofs_global_to_local(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                         std::vector<std::int64_t>& global_dofs);

/// Create an function space with an extended index map, where all input dofs
/// (global index) is added to the local index map as ghosts.
/// @param[in] V The original function space
/// @param[in] global_dofs The list of master dofs (global index)
/// @param[in] owners The owners of the master degrees of freedom
dolfinx::fem::FunctionSpace create_extended_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::vector<std::int64_t>& global_dofs, std::vector<std::int32_t>& owners);

/// Send masters, coefficients, owners and offsets to the process owning the
/// slave degree of freedom.
/// @param[in] master_to_slave MPI neighborhood communicator from processes with
/// master dofs to those owning the slave dofs
/// @param[in] num_remote_masters Number of masters that will be sent to each
/// destination of the neighboorhod communicator
/// @param[in] num_remote_slaves Number of slaves that will be sent to each
/// destination of the neighborhood communicator
/// @param[in] num_incoming_slaves Number of slaves that wil be received from
/// each source of the neighboorhod communicator
/// @param[in] num_masters_per_slave The number of masters for each slave that
/// will be sent to the owning rank
/// @param[in] masters The masters to send to the owning process
/// @param[in] coeffs The corresponding coefficients to send to the owning
/// process
/// @param[in] owners The owning rank of each master
/// @returns Data structure with the number of masters per slave, the master
/// dofs (global indices), the coefficients and owners
template <typename T>
recv_data<T> send_master_data_to_owner(
    MPI_Comm& master_to_slave,
    const std::vector<std::int32_t>& num_remote_masters,
    const std::vector<std::int32_t>& num_remote_slaves,
    const std::vector<std::int32_t>& num_incoming_slaves,
    const std::vector<std::int32_t>& num_masters_per_slave,
    const std::vector<std::int64_t>& masters, const std::vector<T>& coeffs,
    const std::vector<std::int32_t>& owners)
{
  int indegree(-1);
  int outdegree(-2);
  int weighted(-1);
  MPI_Dist_graph_neighbors_count(master_to_slave, &indegree, &outdegree,
                                 &weighted);

  // Communicate how many masters has been found on the other process
  std::vector<std::int32_t> num_recv_masters(indegree);
  MPI_Request request_m;
  MPI_Ineighbor_alltoall(
      num_remote_masters.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      num_recv_masters.data(), 1, dolfinx::MPI::mpi_type<std::int32_t>(),
      master_to_slave, &request_m);

  std::vector<std::int32_t> remote_slave_disp_out(outdegree + 1, 0);
  std::partial_sum(num_remote_slaves.begin(), num_remote_slaves.end(),
                   remote_slave_disp_out.begin() + 1);

  // Send num masters per slave
  std::vector<int> slave_disp_in(indegree + 1, 0);
  std::partial_sum(num_incoming_slaves.begin(), num_incoming_slaves.end(),
                   slave_disp_in.begin() + 1);
  std::vector<std::int32_t> recv_num_masters_per_slave(slave_disp_in.back());
  MPI_Neighbor_alltoallv(
      num_masters_per_slave.data(), num_remote_slaves.data(),
      remote_slave_disp_out.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      recv_num_masters_per_slave.data(), num_incoming_slaves.data(),
      slave_disp_in.data(), dolfinx::MPI::mpi_type<std::int32_t>(),
      master_to_slave);

  // Wait for number of remote masters to be received
  MPI_Status status_m;
  MPI_Wait(&request_m, &status_m);

  // Compute in/out displacements for masters/coeffs/owners
  std::vector<std::int32_t> master_recv_disp(indegree + 1, 0);
  std::partial_sum(num_recv_masters.begin(), num_recv_masters.end(),
                   master_recv_disp.begin() + 1);
  std::vector<std::int32_t> master_send_disp(outdegree + 1, 0);
  std::partial_sum(num_remote_masters.begin(), num_remote_masters.end(),
                   master_send_disp.begin() + 1);

  // Send masters/coeffs/owners to slave process
  std::vector<std::int64_t> recv_masters(master_recv_disp.back());
  std::vector<std::int32_t> recv_owners(master_recv_disp.back());
  std::vector<T> recv_coeffs(master_recv_disp.back());
  std::array<MPI_Status, 3> data_status;
  std::array<MPI_Request, 3> data_request;

  MPI_Ineighbor_alltoallv(
      masters.data(), num_remote_masters.data(), master_send_disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), recv_masters.data(),
      num_recv_masters.data(), master_recv_disp.data(),
      dolfinx::MPI::mpi_type<std::int64_t>(), master_to_slave,
      &data_request[0]);
  MPI_Ineighbor_alltoallv(coeffs.data(), num_remote_masters.data(),
                          master_send_disp.data(), dolfinx::MPI::mpi_type<T>(),
                          recv_coeffs.data(), num_recv_masters.data(),
                          master_recv_disp.data(), dolfinx::MPI::mpi_type<T>(),
                          master_to_slave, &data_request[1]);
  MPI_Ineighbor_alltoallv(
      owners.data(), num_remote_masters.data(), master_send_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), recv_owners.data(),
      num_recv_masters.data(), master_recv_disp.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), master_to_slave,
      &data_request[2]);

  /// Wait for all communication to finish
  MPI_Waitall(3, data_request.data(), data_status.data());
  dolfinx_mpc::recv_data<T> output;
  output.masters = recv_masters;
  output.coeffs = recv_coeffs;
  output.num_masters_per_slave = recv_num_masters_per_slave;
  output.owners = recv_owners;
  return output;
}

} // namespace dolfinx_mpc
