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
   
   MentPy is currently in alpha and is under active development. 

The :obj:`mentpy` library is an open-source Python package for creating and training quantum machine learning (QML) models 
in the measurement-based quantum computing (MBQC) framework. This library contains functions
to automatically calculate the causal flow or generalized flow of a graph and tools to analyze the 
expressivity of the MBQC ansatzes.


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
* Increase code coverage.
* Add autodiff support for MBQC circuits.
* Add support for more general MBQC states.
* Integrate with `pyzx` to optimize resources in MBQC circuits.


Contributing
------------
If you would like to contribute to this project, please feel free to open an issue or pull request 😄.

Acknowledgements
----------------

This library was first developed by Luis Mantilla for his master's thesis
at the University of British Columbia. Luis would like to thank his M.Sc. 
supervisors, Dr. Dmytro Bondarenko, Dr. Polina Feldmann, and Dr. Robert Raussendorf.


Citation
--------

If you find MentPy useful in your research, please consider citing us 🙂

.. md-tab-set::
   .. md-tab-item:: BibTeX

      .. code-block:: latex

         @software{Mantilla_Mentpy_2023,
            title = {{MentPy: A python package for simulating and training QML models in the MBQC framework.}},
            author = {Mantilla, Luis},
            year = {2023},
            url = {https://github.com/bestquark/mentpy},
         }

   .. md-tab-item:: AIP

      .. code-block:: text

         L. Mantilla, MentPy: A python package for simulating and training QML models in the MBQC framework, (2023). https://github.com/bestquark/mentpy
     

   .. md-tab-item:: APA

      .. code-block:: text

         Mantilla, L. (2023). MentPy: A python package for simulating and training QML models in the MBQC framework. Retrieved from https://github.com/bestquark/mentpy
   
   .. md-tab-item:: MLA

      .. code-block:: text

         Mantilla, Luis. MentPy: A python package for simulating and training QML models in the MBQC framework. 2023. Web. https://github.com/bestquark/mentpy



.. toctree::
   :caption: Getting Started
   :hidden:

   getting-started

.. toctree::
   :caption: Basic usage
   :hidden:

   basic-usage/measurements-in-qm.rst
   basic-usage/intro-to-graphstates.rst
   basic-usage/intro-to-mbqc.rst
   basic-usage/simulating-mbqc-circuits.rst

.. toctree::
   :caption: Tutorials
   :hidden:

   tutorials/intro-to-mbqml.rst
   tutorials/intro-to-mbqml-parallel.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api
