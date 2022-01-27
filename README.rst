|licensebuttons by-nc-sa-white|

.. |licensebuttons by-nc-sa-white| image:: https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png
   :target: https://creativecommons.org/licenses/by-nc-sa/4.0/


*dl4HM* - Deep Learning for Hydraulic Modeling
=======================================================

*dl4HM* is a Python package developed to use deep learning for hydraulic modeling. It is based on TensorFlow 2.

- Build surrogate model to hydraulic models such as SRH-2D and HEC-RAS.
- Perform inversion, parameter calibration, and optimization. 

Dependencies
============
The easiest way to install all dependencies is to use the "environment.yml" file and conda:

.. code-block:: bash

    $ conda env create --name dl4HM -f environment.yml

which creates an environment named "dl4HM". You can change the name to suit your need.

Some of the pre- and post-processing functionalities involve the control of hydraulic models such as SRH-2D and HEC-RAS. They depend on the *pyHMT2D* (https://github.com/psu-efd/pyHMT2D) package. This is only needed if you want to use the same scripts to generate training data.

License
-------

Creative Commons Attribution-NonCommercial 4.0 International License


Author
------

| Xiaofeng Liu, Ph.D., P.E.
| Associate Professor

| Department of Civil and Environmental Engineering
| Institute of Computational and Data Sciences
| Penn State University

223B Sackett Building, University Park, PA 16802

Web: http://water.engr.psu.edu/liu/

Contributors and contributor agreement
--------------------------------------
The list of contributors:
^^^^^^^^^^^^^^^^^^^^^^^^^
- (To be added)

Contributor agreement
^^^^^^^^^^^^^^^^^^^^^
First of all, thanks for your interest in contributing to *dl4HM*. Collectively, we can make *dl4HM* more
powerful, better, and easier to use.

Because of legal reasons and like many successful open source projects, contributors have to sign
a "Contributor License Agreement" to grant their rights to "Us". See details of the agreement on GitHub.
The signing of the agreement is automatic when a pull request is issued.

If you are just a user of *dl4HM*, the contributor agreement is irrelevant.
