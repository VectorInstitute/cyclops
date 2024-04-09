Monitoring
==========

Basically, after initial evaluation and report generation, you or your users may have the question that how to monitor model performance over time?
By evaluating the model on the dataset through time and generating reports, you will gain insights about the performance of model over time dimension. Cyclops allows users to select and view the number of latest evaluation.

Everytime an evaluation is performed and added to the report, a new entry is added to the report's `JSON` file and by doing so repeatedly, the users will be able to view the trend of performance over days, months, or hours.

Overview Performance
--------------------

At top level and in a quick glance, there are overall performance metrics:

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/92cacabf-ff8e-42a5-bf3d-f338d0f3ce3d

The number on top left of each figure indicates the metric value for latst evaluation, each with their corresponding timestamp on the X-axis. The figures are color coded base on a minimum threshold that was defined by developers. Once the metric for the latest evaluation drops below the threshold it's shown in red, and when everything is good, it appears in green.

Subgroup Performance
--------------------

To get a better prespective about different subgroups such as age intervals or sex, you have the option of multiple plots in a single figure:

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/9e34e789-8d29-44dc-8631-3d9630fbb8f7


Again, this plot has the feature of viewing a number of past evaluations using the slider at the top right.
