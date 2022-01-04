from IPython import embed
import dolfinx.fem as fem
import dolfinx_mpc.utils
import numpy as np

from dolfinx.common import Timer, list_timings, TimingType
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Identity,
                 SpatialCoordinate,
                 TestFunction,
                 TrialFunction,
                 as_vector,
                 dx,
                 grad,
                 inner,
                 sym,
                 tr
                 )

N = 10
mesh = create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

# Generate Dirichlet BC on lower boundary (Fixed)
u_bc = fem.Function(V)

with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def left_boundary(x):
    return np.isclose(x[0], np.finfo(float).eps)


left_facets = locate_entities_boundary(mesh, 1, left_boundary)
topological_dofs = fem.locate_dofs_topological(V, 1, left_facets)
bc = fem.DirichletBC(u_bc, topological_dofs)
bcs = [bc]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Elasticity parameters
E = PETSc.ScalarType(1.0)
nu = 0.3
mu = fem.Constant(mesh, E / (2.0 * (1.0 + nu)))
lmbda = fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

# Stress computation


def sigma(v):
    return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v))


x = SpatialCoordinate(mesh)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a_form = inner(sigma(u), grad(v)) * dx
L_form = inner(as_vector((10 * (x[1] - 0.5) ** 2, 0)), v) * dx

# Create MPC for imposing u[1] constant on the bottom face


def right_boundary(x):
    return np.isclose(x[0], 1.0)


right_facets = locate_entities_boundary(mesh, 1, right_boundary)
rigid_dofs = fem.locate_dofs_topological(V, 1, right_facets)


def create_rigid_constraint(mpc, dofs, subspace=None):
    dof_coordinates = mpc.V.tabulate_dof_coordinates()[dofs, :]
    constraint_dict = {}
    for i in range(len(dof_coordinates) - 1):
        constraint_dict[dof_coordinates[i + 1, :].tobytes()] = {dof_coordinates[0, :].tobytes(): 1}
    mpc.create_general_constraint(constraint_dict, subspace, subspace)


def periodic(x):
    x_out = x.copy()
    x_out[1] = 0
    return x_out


def indicator(x):
    return np.logical_and(np.isclose(x[0], 0), x[1] > 1e-13)


mpc_data = dolfinx_mpc.cpp.mpc.create_periodic_constraint(V._cpp_object, indicator, periodic, [], 1)
exit()


with Timer("~RIGID: Initialize MPC"):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    create_rigid_constraint(mpc, rigid_dofs, 0)
    mpc.finalize()

problem = dolfinx_mpc.LinearProblem(a_form, L_form, mpc, bcs=bcs)
solver = problem.solver
u_h = problem.solve()
u_h.name = "u"
outfile = XDMFFile(MPI.COMM_WORLD, "rigid_boundary.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(u_h)

list_timings(MPI.COMM_WORLD, [TimingType.wall])
