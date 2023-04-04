Custom SparkML Pipelines
========================

This document includes an example of how to build a custom Estimator and Transformer using the base classes in this repository, and how to integrate them with SparkML Pipelines. For information about the SparkML Pipelines concepts and use of existing Estimators and Transformers within the SparkML module, please refer to the `Spark ML Pipelines <https://spark.apache.org/docs/latest/ml-pipeline.html>`__ documentation.

Build a custom Transformer
--------------------------

In this section we build a Transformer that adds a constant to a column and updates the column's values in-place.

.. note::

    The arguments set in the `__init__` method will be accessible in the `_transform` method as attributes by adding an underscore prefix to the argument name. In the example below the arguments are `column_name` and `value`, and they are available as attributes in the `_transform` method as `self._column_name` and `self._value`.

.. code-block:: python

    import pyspark.sql.functions as F
    from pyspark import keyword_only
    from sparkml_base_classes import TransformerBaseClass


    class AdditionColumnTransformer(TransformerBaseClass):

        @keyword_only
        def __init__(self, column_name=None, value=None):
            super().__init__()

        def _transform(self, ddf):
            self._logger.info("AdditionColumn transform with column {self._column_name}")
            ddf = ddf.withColumn(self._column_name, F.col(self._column_name) + self._value)
            return ddf

Build a custom Estimator
------------------------

In this section we build an Estimator that normalizes the values of a column by the mean. An Estimator's `_fit` method must return a Transformer because the use of an Estimator consists of 2 steps:

1. Fitting the estimator.
    This step consists of using the `_fit` method to calculate some value(s) from the DataFrame and return a Transformer that stores the calculated value(s) and use them in the `_transform` method to transform a DataFrame. In this example the Estimator calculates the mean and returns a Transformer that divides the column by this mean value.

2. Transforming the DataFrame.
    Once the Estimator has been fitted and a Transformer has been returned, then we use the returned Transformer to transform the DataFrame. In this case the Transformer divides the specified column by the mean and returns the transformed DataFrame.

.. note::

    The arguments set in the `__init__` method will be accessible in the `_transform` and `_fit` methods as attributes by adding an underscore prefix to the argument name. In the example below the arguments are `column_name` and `mean`, and they are available as attributes in the `_transform` and `_fit` method as `self._column_name` and `self._mean`.

.. code-block:: python

    import pyspark.sql.functions as F
    from pyspark import keyword_only
    from sparkml_base_classes import EstimatorBaseClass, TransformerBaseClass

    class MeanNormalizerTransformer(TransformerBaseClass):

        @keyword_only
        def __init__(self, column_name=None, mean=None):
            super().__init__()

        def _transform(self, ddf):
            # add your transformation logic here
            self._logger.info("MeanNormalizer transform")
            ddf = ddf.withColumn(self._column_name, F.col(self._column_name) / self._mean)
            return ddf

    class MeanNormalizerEstimator(EstimatorBaseClass):

        @keyword_only
        def __init__(self, column_name=None):
            super().__init__()

        def _fit(self, ddf):
            # add your transformation logic here
            self._logger.info("MeanNormalizer fit")
            mean, = ddf.agg(F.mean(self._column_name)).head()
            return MeanNormalizerTransformer(
                column_name=self._column_name,
                mean=mean
            )

Build the Pipeline
------------------

In this section we will build a Pipeline containing our custom Transformer and Estimator. We will first initialize both classes and then add them as stages to the Pipeline.

.. note::
    We can also use Transformers and Estimators individually by calling their respective `_transform` and `_fit` methods, the advantage of using a Pipeline is to chain them together therefore reducing the code maintenance needed. In addition, it is a good practice to always use them as part of a Pipeline.


.. code-block:: python

    from pyspark.ml import Pipeline

    multiply_column_transformer = AdditionColumnTransformer(column_name="foo", value=2)
    mean_normalizer_estimator = MeanNormalizerEstimator(column_name="foo")
    my_pipeline = Pipeline(stages=[multiply_column_transformer, mean_normalizer_estimator])

Fit the Pipeline and transform the DataFrame
--------------------------------------------

In this section we will fit the created Pipeline to a DataFrame and then use the fitted Pipeline (or PipelineModel in SparkML terms) to transform a DataFrame. Thus, after a Pipeline’s fit method runs, it produces a PipelineModel, which is a Transformer. This PipelineModel can be later used to transform any DataFrame. Please refer to the `Spark ML Pipelines <https://spark.apache.org/docs/latest/ml-pipeline.html#how-it-works>`__ documentation for an in-depth description.

.. note::
    After fitting a Pipeline, the stages containing an Estimator will now contain the Transformer returned in the Estimator's `_fit` method.

.. note::
    The returned object of fitting a Pipeline is not a Pipeline object but a PipelineModel.

.. code-block:: python

    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline

    spark = SparkSession.builder.getOrCreate()

    ddf = spark.createDataFrame(
        [[1], [2], [3]],
        ["foo"],
    )

    # the returned object is of PipelineModel type
    my_fitted_pipeline = my_pipeline.fit(ddf)
    my_fitted_pipeline.transform(ddf).show()

    +----+
    | foo|
    +----+
    |0.75|
    | 1.0|
    |1.25|
    +----+

Save and load fitted Pipeline
-----------------------------

In the previous section we transformed the DataFrame immediately after fitting the Pipeline, in this section we will use an intermediary saving mechanism that allows us to decouple the fitting of the Pipeline from the transforming of the DataFrame.

.. note::
    It is a good practice to save the Pipeline using the `.pipeline` extension.

.. note::
    If you are using Spark in an AWS service, like SageMaker, the path to save the model can be an S3 path. This will work out-of-the-box given that the correct permission to read/write to that path are set.

.. code-block:: python

    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    ddf = spark.createDataFrame(
        [[8], [10], [12]],
        ["foo"],
    )

    my_fitted_pipeline.save('my_fitted_pipeline.pipeline')
    my_fitted_pipeline = PipelineModel.load('my_fitted_pipeline.pipeline')
    my_fitted_pipeline.transform(ddf).show()

    +----+
    | foo|
    +----+
    | 2.5|
    |   3|
    | 3.5|
    +----+

