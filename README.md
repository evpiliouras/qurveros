<div align="left">
<img src="./docs/media/logo/Qurveros_Logo_FULL_white_outline.svg" 
width="320" alt="qurveros logo" >
</img>
</div>


If you share our love for curves on the robust quantum control problem,
this package will allow you to seamlessly experiment with our formalism and 
generate your own unique robust quantum control protocols.

If Space Curve Quantum Control (SCQC) already sounds familiar, you can go ahead 
and install `qurveros`. If you haven't seen SCQC before, you can consult 
the following papers:

&rarr; [An automated geometric space curve approach for designing 
dynamically corrected gates](https://www.nature.com/articles/s41534-026-01190-6)

&rarr; [Dynamically corrected gates from geometric space curves](https://iopscience.iop.org/article/10.1088/2058-9565/ac4421)

# Suppressing noise with space curves
The best starting point in experimenting with qurveros is to consult
the `examples` folder in sequence:
1. `circle_constant_pulse`
2. `bessel_curve_robust_pulse`
3. `circle_optimization`

The package provides a streamlined flow, from the choice of the curve ansatz up
to the extraction of the robust pulses and their simulation. The implementation 
uses JAX to execute the required differentiation and evaluation of 
the functions in a discrete set of points. 

The curve is defined as a function of the form:

```python
def curve(x, params):
        
        x_comp = f(x, params)
        y_comp = g(x, params)
        z_comp = h(x, params)

        return [x_comp, y_comp, z_comp]
```
where `x` is the curve's parameter used to traverse the curve and `params` 
are auxiliary parameters that control its shape or its properties.

The `Spacecurve` class is the heart of the package. An instance is created as:
```python
from qurveros.spacecurve import SpaceCurve

spacecurve = SpaceCurve(curve=curve,
                        order=order, 
                        interval=[x_0, x_1],
                        params=params)
```
The user can provide either the position vector, which is typically referred to
as "curve" and corresponds to `order=0` or the tangent vector with `order=1`, 
in any parametrization. The `interval` of the curve parameter `x` 
with $x \in [x_0, x_1]$ is used to evaluate all the geometric quantities 
in the interval of the given endpoints.
Auxiliary parameters are passed in the last argument.

To perform the required computations, we use:
```python
spacecurve.evaluate_frenet_dict()
spacecurve.evaluate_robustness_properties()
```

where the `frenet_dict` contains all the geometric quantities necessary for 
SCQC and an instance of the class `RobustnessProperties` evaluates the 
robustness properties of the quantum evolution encoded in the properties of 
the curve.

The entries of the frenet dictionary are evaluated at equidistant samples, 
with the default number found in `qurveros.settings.options['CURVE_POINTS']`, 
which can be modified at runtime. Alternatively, the number of desired points 
can be supplied as an argument when the method is called.

For the quantum control problem, we assume a Hamiltonian of the form:

$$
H_0(t) = \frac{\Omega(t)}{2}[\cos\Phi(t) \sigma_x + \sin\Phi(t) \sigma_y] +
 \frac{\Delta(t)}{2}\sigma_z.
$$

Depending on the `control_mode` string that specifies the control axes, we use 
the instance method:

```python
spacecurve.evaluate_control_dict(control_mode)
```
where the `control_dict` attribute contains the control fields based on SCQC. 
The available control modes can be found in `qurveros.controltools`. 
For instance, if resonant control is required, we provide 
the `control_mode` as `'XY'`.

For easy access to qubit simulation tasks, the helper package `qubit_bench` 
automates some of the experiments commonly used to assess the robustness 
properties of the derived pulses. Demonstrations on how to use the 
functionalities of the subpackage is found in the `examples` folder. 
Quantum operations and simulations are handled with `qutip`.

# Purpose and vision
While the package's primary purpose is to serve as an implementation of SCQC 
and BARQ, it also offers a unique opportunity to exist as a guide 
for the newcomers. 

The natural evolution of a package is to expand the set of provided 
functionalities and grow larger and larger over the years.
While the source code of a package constitutes one of the most informative 
means to draw inspiration from and adopt the best programming practices, 
in the eyes of the uninitiated, the vast amount of available information can
make the process feel intimidating.

Given qurveros' small scale, and inspired from the didactic character of 
numerous example sections of the existing packages, I deeply hope that this
work will provide the learner with a simple use-case to start with.

# Contributing
I would be more than happy to receive contributions to this project in any form;
with improvements that regard functionality implementations, documentation and 
adoption of better programming practices. 

In your contributions, please keep in mind the package's vision. Like Pikachu, 
qurveros can still deliver a framework to design robust quantum control 
protocols using space curves, while maintaining an educational character 
to inspire the newcomers.

# Citation
If you found this repository useful to your work, please consider citing the 
associated paper:

&rarr; [An automated geometric space curve approach for designing 
dynamically corrected gates](https://www.nature.com/articles/s41534-026-01190-6)

# Contact
If you want to provide feedback 
(which is strongly encouraged, and is of any sign and magnitude),
please don't hesitate to reach out!

Email: [piliouras[at]vt.edu](mailto:piliouras@vt.edu)

# Appendix: Preparing your quantum canvas
This section provides the necessary steps to setup qurveros in your system.

## Virtual environment
The use of a virtual environment is strongly recommended. Assuming that `python`
is in the path, you can create your own virtual environment using:
```
python3 -m venv .venv
```
which creates a virtual environment `.venv`.

You can activate it using:
```
source .venv/bin/activate
```
For more information with platform-dependent instructions: 
https://docs.python.org/3/library/venv.html

Note: In case your virtual environment is not detected, you will have to add
the interpreter path manually.

## Installation
The package is uploaded on PyPI. For the core functionalities, 
you can use:
```
pip install qurveros
```

In order to run the examples, you can add the required packages using:
```
pip install qurveros[examples]
```

and in order to reproduce the plots in the paper:

```
pip install qurveros[results]
```

For the latest updates, you can install the package by cloning the repository
and using the command (assuming the current working directory contains the
pyproject.toml file):
```
pip install ."[dependencies]"
```

# Acknowledgements
I would like to deeply thank Dennis Lucarelli, Hisham Amer and Kyle Sherbert 
for testing qurveros and providing extremely helpful suggestions 
for the package.