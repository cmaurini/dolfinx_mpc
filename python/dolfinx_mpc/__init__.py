# Copyright (C) 2020 Jørgen Schartum Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    MIT
"""Main module for DOLFINX_MPC"""

# flake8: noqa

import dolfinx_mpc.cpp

# New local assemblies
from .assemble_matrix import assemble_matrix
from .assemble_vector import assemble_vector, apply_lifting
from .multipointconstraint import MultiPointConstraint
from .problem import LinearProblem
