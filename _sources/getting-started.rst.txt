Installation
=============

The :doc:`getting-started` guide is intended to assist the user with installing the library.

Install using ``pip``
---------------------
The :obj:`mentpy` library requires Python 3.9 and above. It can be installed from 
`PyPI <https://pypi.org/project/mentpy/>`_ using ``pip``.

.. code-block:: console

   $ python3 -m pip install mentpy

Install from Source
-------------------

To insall from Source, you can ``git clone`` the repository with

.. code-block:: console

   $ git clone https://github.com/BestQuark/mentpy
   $ cd mentpy
   $ python3 -m pip install -r requirements.txt
   $ python3 setup.py


Import the :obj:`mentpy` package and verify that it was installed correctly.

.. ipython:: python

   import mentpy as mp
   mp.__version__



