**********
Developers
**********
.. highlight:: console


Code guidelines
===============
todo
* pycharm community
* Napoleon style docstring
*


Documentation
=============
In MDT we use the ReStructedText format (extension ``.rst``) for the documentation and we use Sphinx with the Napoleon style docstring for the API documentation generation.
Care must be taken that the docstrings in the Python source code use the Napoleon documentation style.


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
