####################
Developer guidelines
####################
This chapter first contains some helping information on debugging your CL code, with afterwards some coding guidelines for MDT (core-)developers.


.. _debugging_opencl:

****************
Debugging opencl
****************

Evaluating models
=================
To aid debugging of MDT compartment models, composite models and library functions, MDT allows you to evaluate your CL model based on some simple inputs.
For example::

    stick = mdt.get_component('compartment_models', 'Stick')()
    result = stick.evaluate({'g':     [1, 0, 0],
                             'b':     1e9,
                             'd':     1e-9,
                             'theta': 1,
                             'phi':   1
                             }, 1)

In this example we loaded the ``Stick`` compartment from the MDT repository, instantiated it and evalauted it based on a dictionary of input elements.
This :meth:`~mot.lib.cl_function.CLFunction.evaluate` method allows takes as input a dictionary of values (one for each function parameter) and executes the CL code based on those inputs.
If more than one value is given per parameter, the code will be evaluated multiple times, once for each set of parameters.
For example::

    stick = mdt.get_component('compartment_models', 'Stick')()
    result = stick.evaluate({'g':     [1, 0, 0],
                             'b':     1e9,
                             'd':     [1e-9, 2e-9],
                             'theta': [1, 2],
                             'phi':   1
                             }, 2)

In this example, the Stick model will be evaluated twice, first on the set of parameters::

    {'g':     [1, 0, 0],
     'b':     1e9,
     'd':     1e-9,
     'theta': 1,
     'phi':   1
     }


and second, on the set of parameters::

    {'g':     [1, 0, 0],
     'b':     1e9,
     'd':     2e-9,
     'theta': 2,
     'phi':   1
     }


Using the evaluate function, you can also evaluate library functions::

    component = mdt.get_component('library_functions',
                                  'SphericalToCartesian')
    model = component()
    result = model.evaluate({'theta': 0.5, 'phi': 0.5}, 1)


and composite models::

    model = mdt.get_model('BallStick_r1')()
    model_function = model.get_composite_model_function()
    retval = model_function.evaluate({
        'g': [1, 0, 0],
        'b': 2e9,
        'S0.s0': 1000,
        'w_ball.w': 0.5,
        'w_stick0.w': 0.5,
        'Ball.d': 1e-9,
        'Stick0.d': 1e-9,
        'Stick0.theta': 0.5,
        'Stick0.phi': 0.5
    }, 1)


Using the printf function
=========================
In addition to the above, it is also possible to print the execution status within a CL kernel, using the OpenCL ``printf`` command.
The ``printf`` command is part of the OpenCL language and allows you to print some variables during kernel execution.

As an example, suppose we want to print the output of the Stick compartment during model execution.
The original Stick cl code is::

    cl_code = '''
        return exp(-b * d * pown(
            dot(g, SphericalToCartesian(theta, phi)), 2));
    '''

and we want to include printing of the dot product. We then change the code to read::

    cl_code = '''
        printf("%f", dot(g, SphericalToCartesian(theta, phi)));

        return exp(-b * d * pown(
            dot(g, SphericalToCartesian(theta, phi)), 2));
    '''

now, the value of the dot product will be printed during kernel execution.

Please be aware that this may print A LOT of output.
That is, including the above print statement and running the BallStick model on a diffusion dataset will print a value for every voxel, for every volume and for every iteration of the optimization routine.
This can slightly be prevented by providing a mask in which only a single voxel is selected.

For the reference guide on ``printf``, please see: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/printfFunction.html



********************
Developer guidelines
********************
MDT has a few small guidelines to make future collaboration as easy as possible by maintaining code consistency.
Since MDT is written in two languages, Python and OpenCL we have guidelines for both languages.


Variable naming
===============
In general, in both OpenCL and Python, try to use semantically informative names for your functions and variables.
For example, instead of typing:

.. code-block:: python

    def sph2cart(theta, phi):
        st = np.sin(theta)
        sp = np.sin(phi)
        ...

use something like this instead:

.. code-block:: python

    def spherical_to_cartesian(theta, phi):
        sin_theta = np.sin(theta)
        sin_phi = np.sin(phi)
        ...

Here, both the name of the function ``spherical_to_cartesian`` and the names of the variables ``sin_theta``, ``sin_phi``, make it immediately clear what
the function does or what the variables contain.
More in general, avoid acronyms where possible.


Syntax guideline
================
For parts programmed in OpenCL you can primarily use your own syntax style, yet we do prefer that the opening brackets are on the same line as
the function or ``if`` statement, and that the closing brackets are on their own line. For example:

.. code-block:: c

    void my_function(){
        if(...){

        }
        else{

        }
    }

For the Python parts, please follow the general PEP guidelines where possible.
For example, try to not extend the Python code beyond 80 characters.
Also try to avoid the ``... if ... else ...`` style of programming.


*************
Documentation
*************
In MDT we use the ReStructedText format (extension ``.rst``) for the documentation and we use Sphinx with the Napoleon style docstring for the API documentation generation.

For the section headers in the documentation, please follow this convention:

* % with overline, for main title
* # with overline, for parts
* \* with overline, for chapters
* =, for sections
* -, for subsections
* ^, for subsubsections
* ", for paragraphs


Generate the documentation
==========================
Generating the documentation on your workstation is easy using the command ``make docs``.
This command uses Sphinx to generate the documentation from the Python code (the API documentation), and then links it to general documentation files in the ``docs`` directory.
Please note that you will only need to run this command if you want to generate the documentation on your computer, the online MDT documentation is generated automatically.

In order to run the command ``make docs``, you will need to have a few packages installed. To do so, please run:

.. code-block:: bash

    $ sudo apt install python3-pip python3-numpy python3-yaml \
        python3-matplotlib python3-scipy python3-nibabel \
        python3-argcomplete
    $ sudo pip3 install tatsu sphinx alabaster sphinx-argparse sphinxcontrib-bibtex gitchangelog pystache

Some of these commands are Debian/Ubuntu specific, for other operating systems please lookup the corresponding packages for your system.

If you additionally want to generate the PDF documentation file you will have to install some Latex packages as well:

.. code-block:: bash

    $ sudo apt install \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended texlive-latex-extra \
        latexmk


Generating a release (only possible with access rights)
=======================================================
Use ``make prepare-release`` to prepare the release, then use ``make release`` to push the release.

Required packages:

.. code-block:: bash

    $ sudo apt install \
        python3-pystache \
        dput \
        python3-stdeb \
        devscripts \
        build-essential \
        twine

    $ sudo pip3 install \
        gitchangelog \
        python3-wheel
