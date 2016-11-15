**********
Developers
**********
All contributions are welcome to MDT, be it Python code, documentation updates or bug reports.
As with any software project, there are a few guidelines to follow when developing for MDT.
These guidelines are meant to ensure that the code and documentation of MDT stay coherent over time.


Code guidelines
===============
MDT is written in two languages, Python and OpenCL. We have a general naming guideline and for both Python and OpenCL we have a syntax style guideline.


Variable naming
^^^^^^^^^^^^^^^
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
the function does or what the variables contain. More in general, avoid acronyms and shortenings where possible.


Syntax guideline
^^^^^^^^^^^^^^^^
For parts programmed in OpenCL you can primarily use your own syntax style, yet we do prefer that the opening brackets are on the same line as
the function statement or the ``if`` statement and that the closing brackets are on their own line. For example:

.. code-block:: c

    void my_function(){
        if(...){

        }
        else{

        }
    }

For the Python parts, please follow the general PEP guidelines where possible.
For example, try to not extend the Python code beyond 80 characters and try to avoid the ``... if ... else ...`` style of programming.
For example, do not use this style:

.. code-block:: python

    def how_not_to(flag):
        return 0 if flag else 1

but do something like this instead:

.. code-block:: python

    def how_to(flag):
        if flag:
            return 0
        else:
            return 1

or, in this example, this is also fine:

.. code-block:: python

    def how_to_also(flag):
        if flag:
            return 0
        return 1


Documentation
=============
In MDT we use the ReStructedText format (extension ``.rst``) for the documentation and we use Sphinx with the Napoleon style docstring for the API documentation generation.


Generate the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
Generating the documentation on your workstation is easy using the command ``make docs``.
This command uses Sphinx to generate the documentation from the Python code (the API documentation), and then links it to general documentation files in the ``docs`` directory.
Please note that you will only need to run this command if you want to generate the documentation on your computer, the online MDT documentation is generated automatically.

In order to run the command ``make docs``, you will need to have a few packages installed. To do so, please run:

.. code-block:: bash

    $ sudo apt install python3-pip python3-numpy python3-yaml \
        python3-matplotlib python3-scipy python3-nibabel python3-argcomplete
    $ sudo pip3 install grako sphinx alabaster

Some of these commands are Debian/Ubuntu specific, for other operating systems please lookup the corresponding packages for your system.
