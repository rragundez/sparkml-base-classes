"""SparkML base classes for Transformers and Estimators"""

import inspect
import logging
from abc import ABCMeta, abstractmethod
from pyspark import keyword_only
from pyspark.ml import Estimator, Transformer
from pyspark.ml.param import Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable


class _SparkMLMeta(ABCMeta):
    """Meta class to check that all arguments in the constructor have a default value"""

    def __new__(mcs, name, bases, attrs):  # pylint: disable=arguments-differ
        # only check for classes that inherit from the base classes in this module
        if not name.endswith("BaseClass"):
            init_func = attrs.get("__init__", None)
            if init_func is None:
                raise LookupError(f"You must declare an __init__ method for class {name}")
            for base_cls, method in [
                ("EstimatorBaseClass", "_fit"),
                ("TransformerBaseClass", "_transform"),
            ]:
                if len(bases) and (bases[0].__name__ == base_cls) and (method not in attrs):
                    raise LookupError(f"You must declare an {method} method for class {name}")
            init_signature = inspect.signature(init_func)
            for param in init_signature.parameters.values():
                if (param.name != "self") and (param.default == inspect.Parameter.empty):
                    raise SyntaxError(
                        f"Every argument of the {name}.__init__ method must have a default "
                        f"value. Argument {param.name} does not have a default value. "
                        "Remember that you can also set the default value to None."
                    )
            lines = inspect.getsource(init_func)
            if "super().__init__()" not in lines:
                raise SyntaxError(
                    f"The __init__ method of {name} must contain the call to super().__init__()"
                )
            if "@keyword_only" not in lines:
                raise SyntaxError(
                    f"The __init__ method of {name} must use the @pyspark.keyword_only decorator"
                )
        return super().__new__(mcs, name, bases, attrs)


class _SparkMLBaseClass(DefaultParamsReadable, DefaultParamsWritable, metaclass=_SparkMLMeta):
    """Base class to use for Transformers and Estimators base classes"""

    def __init__(self):
        super().__init__()
        # retrieve the child's __init__ frame
        child_frame = inspect.getouterframes(inspect.currentframe())[1].frame
        _set_spark_ml_parameters(self, child_frame)

    @keyword_only
    def setParams(self, **kwargs):  # pylint: disable=invalid-name
        """Set parameters as required by the SparkML standards"""
        return self._set(**self._input_kwargs)  # pylint: disable=no-member

    @property
    def _logger(self):
        """Logger to use within the child SparkML Transformer"""
        return logging.getLogger(self.__class__.__name__)


def _set_spark_ml_parameters(class_instance, init_frame):
    """Assign class arguments as instance parameters according to SparkMl standards.

    Args:
        class_instance: Instantiated class.
        init_frame: Frame belonging to the __init__ function of the class.
    """
    # get child class and extract the signature of the __init__ constructor method
    cls = inspect.getargvalues(init_frame).locals["__class__"]
    init_signature = inspect.signature(cls.__init__)
    for param in init_signature.parameters.values():
        # use arguments in the child class __init__ method expect for self
        if param.name != "self":
            # follow SparkML standards to assign a parameter to the class
            setattr(class_instance, param.name, Param(class_instance, param.name, ""))
            if param.default == inspect.Parameter.empty:
                raise TypeError(
                    f"Every argument of the {cls.__qualname__}.__init__ function must have a "
                    f"default value. Argument {param.name} does not have a default value. "
                    "Remember that you can also set the default value to None."
                )
            # pylint: disable=protected-access
            class_instance._setDefault(**{param.name: param.default})
            # create property getters for all attributes by adding an underscore as prefix
            setattr(
                cls,
                f"_{param.name}",
                property(lambda s, param_name=param.name: s.getOrDefault(getattr(s, param_name))),
            )
    # set parameters as required by the SparkML standards
    class_instance.setParams(**class_instance._input_kwargs)  # pylint: disable=protected-access


class EstimatorBaseClass(_SparkMLBaseClass, Estimator):  # pylint: disable=too-many-ancestors
    """This class handles a lot of the overhead code needed to make a custom SparkML Transformer.

    Taking this example `in stackoverflow <https://stackoverflow.com/a/37279526>`__,
    this base class handles the following parts of the code:

        - The instantiation of each argument as a SparkML Param.
        - Setting the default value for each argument according to SparkML standards.
        - Create the `setParams` method.
        - Create the property method wrappers around `self.getOrDefault` for each argument
          according to SparkML standards.
        - Expose the `_logger` property to use for logging messages inside the class.

    This allows you to create a SparkML Transformer by simply extending from this class and
    defining a `_fit` method.

    It is important to note, that the arguments set in the child's __init__ constructor method
    will be accessible as attributes by adding an underscore prefix to the argument name.
    Below an example of how to use this base class and the use of the attributes inside the
    `_transform` method.

    Example:
        .. code-block:: python

            import pyspark.sql.functions as F
            from pyspark import keyword_only
            from pyspark.sql import SparkSession
            from sparkml_base_classes import EstimatorBaseClass, TransformerBaseClass

            class MeanNormalizerTransformer(TransformerBaseClass):

                @keyword_only
                def __init__(self, column_name=None, mean=None):
                    super().__init__()

                def _transform(self, ddf):
                    self._logger.info("MeanNormalizer transform")
                    ddf = ddf.withColumn(self._column_name, F.col(self._column_name) / self._mean)
                    return ddf

            class MeanNormalizerEstimator(EstimatorBaseClass):

                @keyword_only
                def __init__(self, column_name=None):
                    super().__init__()

                def _fit(self, ddf):
                    self._logger.info("MeanNormalizer fit")
                    mean, = ddf.agg(F.mean(self._column_name)).head()
                    return MeanNormalizerTransformer(
                        column_name=self._column_name,
                        mean=mean
                    )

            spark = SparkSession.builder.getOrCreate()

            ddf = spark.createDataFrame(
                [[1], [2], [3]],
                ["foo"],
            )
            mean_normalizer = MeanNormalizerEstimator(column_name="foo").fit(ddf)
            ddf = mean_normalizer.transform(ddf)
            ddf.show()

            +---+
            |foo|
            +===+
            |0.5|
            +---+
            |1.0|
            +---+
            |1.5|
            +---+

            ddf = spark.createDataFrame(
                [[4], [6], [8]],
                ["foo"],
            )
            ddf = mean_normalizer.transform(ddf)
            ddf.show()

            +---+
            |foo|
            +===+
            |2.0|
            +---+
            |3.0|
            +---+
            |4.0|
            +---+
    """

    @abstractmethod
    def _fit(self, ddf):  # pylint: disable=arguments-renamed
        """The child class must overwrite this method or an exception will be raised"""


class TransformerBaseClass(_SparkMLBaseClass, Transformer):  # pylint: disable=too-many-ancestors
    """This class handles a lot of the overhead code needed to make a custom SparkML Transformer.

    Taking this example `in stackoverflow <https://stackoverflow.com/a/32337101>`__,
    this base class handles the following parts of the code:

        - The instantiation of each argument as a SparkML Param.
        - Setting the default value for each argument according to SparkML standards.
        - Create the `setParams` method.
        - Create the property method wrappers around `self.getOrDefault` for each argument
          according to SparkML standards.
        - Expose the `_logger` property to use for logging messages inside the class.

    This allows you to create a SparkML Transformer by simply extending from this class and
    defining a `_transform` method.

    It is important to note, that the arguments set in the child's __init__ constructor method
    will be accessible as attributes by adding an underscore prefix to the argument name.
    Below an example of how to use this base class and the use of the attributes inside the
    `_transform` method.

    Example:
        .. code-block:: python

            import pyspark.sql.functions as F
            from pyspark import keyword_only
            from pyspark.sql import SparkSession
            from sparkml_base_classes import TransformerBaseClass


            class AddConstantColumnTransformer(TransformerBaseClass):

                @keyword_only
                def __init__(self, column_name=None, value=None):
                    super().__init__()

                def _transform(self, ddf):
                    self._logger.info("AddConstantColumn transform")
                    ddf = ddf.withColumn(self._column_name, F.lit(self._value))
                    return ddf

            spark = SparkSession.builder.getOrCreate()

            ddf = spark.createDataFrame(
                [[1], [2], [3]],
                ["id"],
            )
            add_custom_column_transformer = AddConstantColumnTransformer(
                column_name="bar",
                value="foo"
            )
            ddf = add_custom_column_transformer.transform(ddf)
            ddf.show()

            +---+---+
            | id|bar|
            +===+===+
            |  1|foo|
            +---+---+
            |  2|foo|
            +---+---+
            |  3|foo|
            +--+---+
    """

    @abstractmethod
    def _transform(self, ddf):  # pylint: disable=arguments-renamed
        """The child class must overwrite this method or an exception will be raised"""
