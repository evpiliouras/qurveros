"""
This module provides helper functions to implement BARQ.
"""
import jax
import jax.numpy as jnp

from qurveros import beziertools, frametools
from qurveros.settings import settings


def make_barq_fun(adj_target, pgf_mod, prs_fun):

    """
    Create the function that maps free points, modifications of the PGF and
    structure from the PRS to a set of control points that implement
    the BARQ method. The adjoint representation of the target gate is
    required for the gate-fixing process.

    Args:
        adj_target (array): A 3x3 matrix which corresponds to the adjoint
        representation of the target SU(2) gate.
        pgf_mod (function): A function that modifies the PGF.
        prs_fun (function): A function that enforces a structure in the
        control points.
        See optspacecurve.BarqCurve for more details.

    Returns:
        The barq function that provides the control points
        for the Bezier curve.
    """

    if pgf_mod is None:

        def pgf_mod(pgf_params,  input_points):
            return pgf_params

    if prs_fun is None:

        def prs_fun(prs_params,  input_points):
            return input_points[6:, :]

    @jax.jit
    def barq_fun(params):

        pgf_params = params['pgf_params']
        pgf_params = pgf_mod(pgf_params, params['free_points'])

        gate_fixing_points = point_gate_fixing(params['free_points'][:2, :],
                                               pgf_params,
                                               adj_target)

        input_points = jnp.vstack([
            gate_fixing_points,
            params['free_points'][2:, :]
        ])

        internal_points = prs_fun(params['prs_params'], input_points)

        barq_points = jnp.vstack([
            # Closed curve condition
            jnp.zeros(params['free_points'].shape[1]),
            gate_fixing_points[:3, :],
            internal_points,
            gate_fixing_points[3:, :],
            # Closed curve condition
            jnp.zeros(params['free_points'].shape[1])
        ])

        return barq_points

    return barq_fun


@jax.jit
def point_gate_fixing(initial_points, pgf_params, adj_target):

    r"""
    Implements the arrangement of the control points, at the point-gate fixing
    stage of BARQ.
    The first two points from the free points are free parameters that
    define R_B(x=0).

    The control points of the Bezier curve are set in a way that the curve
    is always closed and the curvature vanishes at the endpoints.

    The remaining end-point control points are set according to the target
    SU(2) provided in the adjoint representation form (adj_target), so that
    the target gate remains encoded in the control points and the optimization
    does not alter the fidelity.

    Args:
        initial_points (array): An 2 x 3 array, which are typically the
        first two free points.
        pgf_params (dict): A dictionary containing the various choices
        available for the scales of the end-point vectors
        and the barq angle $\theta_B$.
        adj_target (array): The target SU(2) gate expressed in the adjoint
        representation.

    Returns:
        An array of dimensions 6 x 3 that encode the target gate, and enforce
        vanishing curvatures at the boundaries.

    Note:
        The initial value of the phase field is assumed to be zero.
    """

    # Enforces the non-zero norm condition for
    # the control points of R_B(t=T_g).

    for fix_param in ['left_tangent_fix', 'left_binormal_fix',
                      'right_binormal_fix', 'right_tangent_fix']:

        pgf_params[fix_param] = jnp.clip(pgf_params[fix_param],
                                         settings.options['FIX_NORM'])

    left_tangent = initial_points[0, :]/frametools.vector_norm(
                   initial_points[0, :])

    left_binormal = initial_points[1, :]/frametools.vector_norm(
                    initial_points[1, :])

    left_binormal = pgf_params['left_binormal_fix']*left_binormal +\
        pgf_params['left_binormal_aux']*left_tangent

    total_rotation = adj_target @ beziertools.bezier_adj(initial_points[0, :],
                                                         initial_points[1, :],
                                                         angle=0)

    right_tangent = -total_rotation[-1, :]

    right_binormal = jnp.sin(pgf_params['barq_angle'])*total_rotation[0, :] +\
        -1*jnp.cos(pgf_params['barq_angle'])*total_rotation[1, :]

    right_binormal = pgf_params['right_binormal_fix']*right_binormal +\
        pgf_params['right_binormal_aux']*right_tangent

    gate_fixing_points = jnp.vstack([
        # Vanishing envelope condition
        pgf_params['left_tangent_fix']*left_tangent,
        pgf_params['left_tangent_aux']*left_tangent,
        left_binormal,
        # Gate fixing Wn-3
        right_binormal,
        # Gate fixing and Vanishing envelope condition Wn-2
        pgf_params['right_tangent_aux']*right_tangent,
        # Gate fixing
        pgf_params['right_tangent_fix']*right_tangent])

    return gate_fixing_points


def get_default_pgf_params_dict():

    r"""
    Creates the pgf_params dictionary with default values.

    For entries with postfix _fix, the values need to be positive and a
    minimum norm is enforced in the barq_fun function.

    The postfix _aux indicates parameters that can take arbitrary values.

    The barq angle $\theta_B$ refers to the angle created locally from
    the orthogonal vectors of the Bezier curve at x=1.
    """

    pgf_params = {}

    frame_vectors = ['tangent', 'binormal']
    vector_locations = ['left', 'right']
    vector_param_types = ['aux', 'fix']

    for vector in frame_vectors:
        for location in vector_locations:
            for param_type in vector_param_types:

                pgf_params[location+'_'+vector+'_'+param_type] = 1.

    pgf_params['barq_angle'] = jnp.pi
    pgf_params['right_binormal_aux'] = 0.
    pgf_params['left_binormal_aux'] = 0.

    return pgf_params
