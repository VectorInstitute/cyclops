Model Report
============

The model report helps technicians, data scientists and clinicians to understand the model's performance better by offering:


    * Clear Visualizations: Effortlessly incorporate the generated figures from model evaluation into your report, providing a clear picture of model performance for everyone.

    .. image:: examples/images/overview_metrics.png

    .. image:: examples/images/overview_performance.png


    * Detailed Model Specs: Document and view all relevant model details and parameters for easy reference.

    .. image:: examples/images/model_details.png

    * Interactive Exploration: Gain insights into model performance across different subgroups over time. Interact with the plots to select specific subgroups and adjust displayed metrics.

    .. image:: examples/images/metrics_comparison.png


Dataset
-------
In dataset section, you will be able to view all the plots that are generated to explore distribution of dataset features. By hovering on any part of the plot you see the detail about that feature. Also, the plots allows interaction such as zooming or panning:

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/85186099-d932-4fe5-8ac6-ee06f4736a3a


Quantitative Analysis
---------------------
Quantitative analysis is somehow a subset of overview, where users are able to further investigate last evaluation results with extra metrics and plots for each slice of dataset.

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/90500d21-94ba-4ede-b488-97669df21a6e


Metric comparison charts are also a handy tool to compare how the model is performing in different subgroups and over all of them.

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/5a5f8300-18de-4737-918e-9d77c33a1ceb


Fairness Analysis
-----------------

Follow the example below for the instructions on how to generate a model report:

.. toctree::

    examples/report.ipynb
