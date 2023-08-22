.. OPA documentation master file, created by
   sphinx-quickstart on Tue Aug 22 21:20:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OPA's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Launch services
==================

Build the images
----------------
Run the script to setup.sh with the argument -b to build all images

.. code-block:: bash

   cd docker
   ./build.sh -b
Local start using docker-compose
--------------------------------

Run the script to setup.sh with the argument -r

.. code-block:: bash

   cd docker
   ./build.sh -r
to stop all processes use the following

.. code-block:: bash

   ./build.sh -s

Deployment with kubernetes
--------------------------
Go to kubernetes folder and launch all the services

.. code-block:: bash

   cd kubernetes
   kubeclt apply -f postgresql
   kubeclt apply -f dashboard

To stop all delete all the deployed services

.. code-block:: bash

   cd kubernetes
   kubeclt delete -f postgresql
   kubeclt delete -f dashboard

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
