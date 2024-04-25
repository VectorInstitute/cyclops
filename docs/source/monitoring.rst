Monitoring
==========

After initial evaluation and model report generation, how can we monitor model
performance over time?

We accomplish this by evaluating the model on new test data and generating new model
reports. We then see the performance trends over time.
The model report allows users to select and view the number of latest evaluations.

Everytime an evaluation is performed and added to the report, a new entry is added
to the model report's `JSON` file and by doing so repeatedly,
the users will be able to view the trend of performance over days, weeks, or months.

.. image:: https://github.com/VectorInstitute/cyclops/assets/8986523/f71cf618-caac-46f7-9221-48d6a71dc1a6

Overview Performance
--------------------

At top level and in a quick glance, there are overall performance metrics:

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/92cacabf-ff8e-42a5-bf3d-f338d0f3ce3d

The number on the top left of each figure indicates the metric value for latest
evaluation, each with their corresponding timestamp on the x-axis. The figures are
color coded based on a minimum threshold that was set by the user.
Once the metric for the latest evaluation drops below the acceptable range it's shown
in red, and when everything is good, it appears in green.

Subgroup Performance
--------------------

To get a better prespective about different subgroups such as age intervals or sex,
you have the option of visualizing multiple plots in a single figure:

.. image:: https://github.com/VectorInstitute/cyclops/assets/5112312/9e34e789-8d29-44dc-8631-3d9630fbb8f7


Again, this plot has the feature of viewing a number of past evaluations using the
slider at the top right.
