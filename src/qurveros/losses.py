"""
This module wraps some frametools functions and converts them to losses for
optimization.
"""

import jax
import jax.numpy as jnp

from qurveros import frametools


@jax.jit
def tantrix_zero_area_loss(frenet_dict):

    area = frametools.calculate_tantrix_area(frenet_dict)

    return jnp.sum(area**2)


@jax.jit
def curve_zero_area_loss(frenet_dict):

    curve_area = frametools.calculate_curve_area(frenet_dict)

    return jnp.sum(curve_area**2)


@jax.jit
def max_amp_loss(frenet_dict):

    r"""
    Calculates the maximum amplitude $T_g \Omega_max$ for minimization.
    """

    total_length = frametools.calculate_total_length(frenet_dict)

    max_amp = total_length*jnp.max(frenet_dict['curvature'])

    return max_amp


@jax.jit
def total_time_loss(frenet_dict):

    total_length = frametools.calculate_total_length(frenet_dict)

    return total_length


@jax.jit
def barq_detuning_loss(frenet_dict):

    r"""
    Creates the loss to minimize the TTC detuning when the BARQ method is used.
    The barq angle $\theta_B$ is an additional degree of freedom that
    enters the optimization in the gate-fixing stage.
    See barqtools.py for more details.
    """

    angle = frenet_dict['params']['pgf_params']['barq_angle'][0]
    ttc_detuning = frametools.calculate_ttc_detuning(frenet_dict, angle)

    return ttc_detuning**2


@jax.jit
def total_curvature_loss(frenet_dict):

    total_curvature = frametools.calculate_total_curvature(frenet_dict)

    return total_curvature


@jax.jit
def cfi_value_loss(frenet_dict):

    return frametools.calculate_cfi_value(frenet_dict)
