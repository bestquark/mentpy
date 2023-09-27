.. MentPy documentation master file, created by
   sphinx-quickstart on Tue Sep  6 11:39:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ./_static/logo.png
   :align: center
   :width: 70%

Welcome to MentPy's documentation
=================================

.. admonition:: Note
   :class: warning
   
   MentPy is currently in alpha and many features are still in development and may not work as expected.

The :obj:`mentpy` library is a Python package for simulating MBQC circuits. This library contains functions
to automatically calculate the causal flow or generalized flow of a graph, and deal with correction and 
byproduct operators in MBQC circuits. 


Features
--------

* Manipulation of graph states.
* Automatically calculate the causal flow or generalized flow of a graph.
* Simulate MBQC circuits.
* Optimize measurement angles in MBQC ansatzes used for QML.
* Create data and noisy data for training QML models.
* Determine the lie algebra of an MBQC ansatz.

Roadmap
-------
* Improve current simulators for MBQC circuits.
* Fix many bugs in the library 🐛.
* Improve tests on current functions.
* Add a tensor network simulator for MBQC circuits.
* Add support for more general MBQC states.
* Integrate with `pyzx` to optimize resources in MBQC circuits.


Contributing
------------
If you would like to contribute to this project, please feel free to open an issue or pull request 😄.

Acknowledgements
----------------

This library is being developed as part of my master's thesis at the University of British Columbia.
I would like to thank my supervisors, Dr. Dmytro Bondarenko, Dr. Polina Feldmann, and Dr. Robert Raussendorf.


Citation
--------

If you find MentPy useful in your research, please consider citing us 🙂

.. md-tab-set::
   .. md-tab-item:: BibTeX

      .. code-block:: latex

         @software{Mantilla_Mentpy_2023,
            title = {{MentPy: A Python library for simulating MBQC circuits}},
            author = {Mantilla, Luis},
            year = {2023},
            url = {https://github.com/bestquark/mentpy},
         }

   .. md-tab-item:: AIP

      .. code-block:: text

         L. Mantilla, MentPy: A Python library for simulating MBQC circuits, (2023). https://github.com/bestquark/mentpy
     

   .. md-tab-item:: APA

      .. code-block:: text

         Mantilla, L. (2023). MentPy: A Python library for simulating MBQC circuits. Retrieved from https://github.com/bestquark/mentpy
   
   .. md-tab-item:: MLA

      .. code-block:: text

         Mantilla, Luis. MentPy: A Python Library for Simulating MBQC Circuits. 2023. Web. https://github.com/bestquark/mentpy



.. toctree::
   :caption: Getting Started
   :hidden:
   :maxdepth: 2

   getting-started

.. toctree::
   :caption: Tutorials
   :hidden:
   :maxdepth: 2

   tutorials/getting-started/measurements-in-qm.rst
   tutorials/getting-started/intro-to-graphstates.rst
   tutorials/getting-started/intro-to-mbqc.rst
   tutorials/getting-started/simulating-mbqc-circuits.rst

   tutorials/mbqml/intro-to-mbqml.rst
   tutorials/mbqml/intro-to-mbqml-parallel.rst
   

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api
