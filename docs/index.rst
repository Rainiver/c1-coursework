Welcome to 5D Interpolator Documentation
=========================================

A PyTorch-based neural network for interpolating 5-dimensional data with FastAPI backend and Next.js frontend.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   usage
   api/index
   testing
   performance

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

This project implements a neural network interpolator for 5D input data, achieving:

* **Test RMSE:** 0.024635
* **R² Score:** 0.9946
* **Architecture:** [256, 128, 64, 32] neurons
* **Training Time:** < 1 second for 5000 samples

Features
--------

* PyTorch-based MLP model with advanced training techniques
* FastAPI REST API backend
* Next.js frontend for data upload, training, and prediction
* Docker deployment with docker-compose
* Comprehensive performance benchmarking

System Requirements
-------------------

* Python 3.9+
* Node.js 20+
* Docker & Docker Compose (for deployment)
* 8GB+ RAM recommended
