# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT


import dolfinx.fem as fem
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg

from dolfinx.common import Timer
from dolfinx.generation import UnitSquareMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from dolfinx_mpc import (MultiPointConstraint, apply_lifting, assemble_matrix,
                         assemble_vector)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, dx, grad, inner, sym, tr)


def demo_elasticity():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 10, 10)

    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Generate Dirichlet BC on lower boundary (Fixed)
    u_bc = fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def boundaries(x):
        return np.isclose(x[0], np.finfo(float).eps)
    facets = locate_entities_boundary(mesh, 1, boundaries)
    topological_dofs = fem.locate_dofs_topological(V, 1, facets)
    bc = fem.DirichletBC(u_bc, topological_dofs)
    bcs = [bc]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Elasticity parameters
    E = PETSc.ScalarType(1.0e4)
    nu = 0.0
    mu = fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * sym(grad(v))
                + lmbda * tr(sym(grad(v))) * Identity(len(v)))

    x = SpatialCoordinate(mesh)
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), grad(v)) * dx
    rhs = inner(as_vector((0, (x[0] - 0.5) * 10**4 * x[1])), v) * dx

    # Create MPC
    with Timer("~Elasticity: Initialize MPC"):
        def l2b(li):
            return np.array(li, dtype=np.float64).tobytes()
        s_m_c = {l2b([1, 0]): {l2b([1, 1]): 0.9}}
        mpc = MultiPointConstraint(V)
        mpc.create_general_constraint(s_m_c, 1, 1)
        mpc.finalize()

    # Setup MPC system
    with Timer("~Elasticity: Assemble LHS and RHS"):
        A = assemble_matrix(a, mpc, bcs=bcs)
        b = assemble_vector(rhs, mpc)
        apply_lifting(b, [a], [bcs], mpc)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    # Write solution to file
    u_h = fem.Function(mpc.function_space)
    u_h.vector.setArray(uh.array)
    u_h.name = "u_mpc"
    outfile = XDMFFile(MPI.COMM_WORLD, "results/demo_elasticity.xdmf", "w")
    outfile.write_mesh(mesh)
    outfile.write_function(u_h)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    A_org = fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = fem.assemble_vector(rhs)
    fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(L_org, bcs)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A_org)
    u_ = fem.Function(V)
    solver.solve(L_org, u_.vector)
    u_.x.scatter_forward()
    u_.name = "u_unconstrained"
    outfile.write_function(u_)
    outfile.close()

    root = 0
    with Timer("~Demo: Verification"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

        # Gather LHS, RHS and solution on one process
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)
        u_mpc = dolfinx_mpc.utils.gather_PETScVector(uh, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d
            assert np.allclose(uh_numpy, u_mpc)

    # Print out master-slave connectivity for the first slave
    master_owner = None
    master_data = None
    slave_owner = None
    if mpc.num_local_slaves > 0:
        slave_owner = MPI.COMM_WORLD.rank
        bs = mpc.function_space.dofmap.index_map_bs
        slave = mpc.slaves[0]
        print("Constrained: {0:.5e}\n Unconstrained: {1:.5e}"
              .format(uh.array[slave], u_.vector.array[slave]))
        master_owner = mpc._cpp_object.owners.links(slave)[0]
        _masters = mpc.masters
        master = _masters.links(slave)[0]
        glob_master = mpc.function_space.dofmap.index_map.local_to_global([master // bs])[0]
        coeffs, offs = mpc.coefficients()
        master_data = [glob_master * bs + master % bs,
                       coeffs[offs[slave]:offs[slave + 1]][0]]
        # If master not on proc send info to this processor
        if MPI.COMM_WORLD.rank != master_owner:
            MPI.COMM_WORLD.send(master_data, dest=master_owner, tag=1)
        else:
            print("Master*Coeff: {0:.5e}".format(coeffs[offs[slave]:offs[slave + 1]][0]
                                                 * uh.array[_masters.links(slave)[0]]))
    # As a processor with a master is not aware that it has a master,
    # Determine this so that it can receive the global dof and coefficient
    master_recv = MPI.COMM_WORLD.allgather(master_owner)
    for master in master_recv:
        if master is not None:
            master_owner = master
            break
    if slave_owner != master_owner and MPI.COMM_WORLD.rank == master_owner:
        bs = mpc.function_space.dofmap.index_map_bs
        in_data = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=1)
        l2g = mpc.function_space.dofmap.index_map.global_indices()
        l_index = np.flatnonzero(l2g == in_data[0] // bs)[0]
        print("Master*Coeff (on other proc): {0:.5e}"
              .format(uh.array[l_index * bs + in_data[0] % bs] * in_data[1]))


if __name__ == "__main__":
    demo_elasticity()
    # list_timings(MPI.COMM_WORLD, [TimingType.wall])
