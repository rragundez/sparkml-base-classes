import logging
import types
import chispa
import pytest
import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.param import Param

from src import sparkml_base_classes as sbc


def test_sparkmlmeta():

    # check that all the methods are added by the metaclass when arguments are None
    class TestClass(metaclass=sbc._SparkMLMeta):
        @keyword_only
        def __init__(self, a=None, b=None):
            super().__init__()

    assert hasattr(TestClass, "__init__")
    assert isinstance(TestClass.__init__, types.FunctionType)
    assert hasattr(TestClass, "__dict__")
    assert hasattr(TestClass, "__module__")
    assert TestClass.__module__ == "sparkml_base_classes_test"

    # check that all the methods are added by the metaclass when argument are not None
    class TestClass(metaclass=sbc._SparkMLMeta):
        @keyword_only
        def __init__(self, a=2, b=1):
            super().__init__()

    assert hasattr(TestClass, "__init__")
    assert isinstance(TestClass.__init__, types.FunctionType)
    assert hasattr(TestClass, "__dict__")
    assert hasattr(TestClass, "__module__")
    assert TestClass.__module__ == "sparkml_base_classes_test"


def test_sparkmlmeta_errors():

    # check that it complains about not having an __init__ method
    with pytest.raises(LookupError, match=r".*__init__.*"):

        class TestClass(metaclass=sbc._SparkMLMeta):
            ...

    # check that it complains about parameters missing a default value
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class TestClass(metaclass=sbc._SparkMLMeta):
            def __init__(self, zz):
                ...

    # check that it complains about parameters missing a default value even if another has
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class TestClass(metaclass=sbc._SparkMLMeta):
            def __init__(self, zz, b=None):
                ...

    # check that it complains if __init__ method is not calling the parent's __init__ method
    with pytest.raises(SyntaxError, match=r".*super.*\.__init()__.*"):

        class TestClass(metaclass=sbc._SparkMLMeta):
            def __init__(self, a=None):
                ...

    # check that it complains if __init__ method is not using the @pyspark.keyword_only decorator
    with pytest.raises(SyntaxError, match=r".*@pyspark\.keyword_only.*"):

        class TestClass(metaclass=sbc._SparkMLMeta):
            def __init__(self, a=None):
                super().__init__()


def test_sparkmlbaseclass():

    # check that all methods are added by base class
    class SomeSparkMLBaseClass(sbc._SparkMLBaseClass):
        def __init__(self, a=None):
            ...

    sparkml_baseclass = SomeSparkMLBaseClass()
    assert hasattr(sparkml_baseclass, "__init__")
    assert isinstance(sparkml_baseclass.__init__, types.MethodType)
    assert hasattr(sparkml_baseclass, "setParams")
    assert isinstance(sparkml_baseclass.setParams, types.MethodType)
    assert hasattr(sparkml_baseclass, "_logger")
    assert isinstance(sparkml_baseclass._logger, logging.Logger)
    assert sparkml_baseclass._logger.name == "SomeSparkMLBaseClass"


class SomeEstimatorThatWorks(sbc.EstimatorBaseClass):
    @keyword_only
    def __init__(self, column_name=None):
        super().__init__()

    def _fit(self, ddf):
        # add your transformation logic here
        self._logger.info("Calculating the mean")
        (mean,) = ddf.agg(F.mean(self._column_name)).head()
        return SomeTransformerThatWorks(column_name=self._column_name, mean=mean)


def test_estimatorbaseclass(tmp_path, spark_session):

    input_ddf = spark_session.createDataFrame([[1], [2], [3]], "foo: int")

    estimator = SomeEstimatorThatWorks()
    assert hasattr(estimator, "__init__")
    assert isinstance(estimator.__init__, types.MethodType)
    assert hasattr(estimator, "setParams")
    assert isinstance(estimator.setParams, types.MethodType)
    assert hasattr(estimator, "_logger")
    assert isinstance(estimator._logger, logging.Logger)
    assert estimator._logger.name == "SomeEstimatorThatWorks"
    assert hasattr(estimator, "column_name")
    assert hasattr(estimator, "_column_name")
    assert isinstance(estimator.column_name, Param)
    assert estimator._column_name is None

    estimator = SomeEstimatorThatWorks(column_name="foo")
    assert hasattr(estimator, "column_name")
    assert hasattr(estimator, "_column_name")
    assert isinstance(estimator.column_name, Param)
    assert estimator._column_name == "foo"
    transformer = estimator.fit(input_ddf)
    assert isinstance(transformer, SomeTransformerThatWorks)
    ddf = transformer.transform(input_ddf)
    test_ddf = spark_session.createDataFrame([[0.5], [1.0], [1.5]], "foo: double")
    chispa.assert_df_equality(ddf, test_ddf, ignore_nullable=True)

    path = tmp_path / "estimator_dir"
    path.mkdir()
    p = Pipeline(stages=[estimator])
    p_fitted = p.fit(input_ddf)
    p_fitted.save(path)
    p_fitted = PipelineModel.load(path)
    input_ddf = spark_session.createDataFrame([[5], [6], [7]], "foo: int")
    ddf = p_fitted.transform(input_ddf)
    test_ddf = spark_session.createDataFrame([[2.5], [3.0], [3.5]], "foo: double")
    chispa.assert_df_equality(ddf, test_ddf, ignore_nullable=True)


def test_estimatorbaseclass_errors():

    # check that it complains about not having an __init__ method
    with pytest.raises(LookupError, match=r".*__init__.*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            ...

    # check that it complains about not having an _fit method
    with pytest.raises(LookupError, match=r".*_fit.*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            def __init__(self):
                ...

    # check that it complains about parameters missing a default value
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            def __init__(self, zz):
                ...

            def _fit(self, ddf):
                ...

    # check that it complains about parameters missing a default value even if another has
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            def __init__(self, zz, b=None):
                ...

            def _fit(self, ddf):
                ...

    # check that it complains if __init__ method is not calling the parent's __init__ method
    with pytest.raises(SyntaxError, match=r".*super.*.__init()__.*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            def __init__(self, a=None):
                ...

            def _fit(self, ddf):
                ...

    # check that it complains if __init__ method is not using the @pyspark.keyword_only decorator
    with pytest.raises(SyntaxError, match=r".*@pyspark\.keyword_only.*"):

        class SomeEstimator(sbc.EstimatorBaseClass):
            def __init__(self, a=None):
                super().__init__()

            def _fit(self, ddf):
                ...


# it needs to be outside the test function in order to test saving and loading
class SomeTransformerThatWorks(sbc.TransformerBaseClass):
    @keyword_only
    def __init__(self, column_name=None, mean=None):
        super().__init__()

    def _transform(self, ddf):
        self._logger.info("Normalizing values by the mean")
        ddf = ddf.withColumn(self._column_name, F.col(self._column_name) / self._mean)
        return ddf


def test_transformerbaseclass(tmp_path, spark_session):

    input_ddf = spark_session.createDataFrame([[1], [2], [3]], "foo: int")

    transformer = SomeTransformerThatWorks()
    assert hasattr(transformer, "__init__")
    assert isinstance(transformer.__init__, types.MethodType)
    assert hasattr(transformer, "setParams")
    assert isinstance(transformer.setParams, types.MethodType)
    assert hasattr(transformer, "_logger")
    assert isinstance(transformer._logger, logging.Logger)
    assert transformer._logger.name == "SomeTransformerThatWorks"
    assert hasattr(transformer, "mean")
    assert hasattr(transformer, "_mean")
    assert isinstance(transformer.mean, Param)
    assert transformer._mean is None
    assert hasattr(transformer, "column_name")
    assert hasattr(transformer, "_column_name")
    assert isinstance(transformer.column_name, Param)
    assert transformer._column_name is None

    transformer = SomeTransformerThatWorks(column_name="foo", mean=2.0)
    assert hasattr(transformer, "mean")
    assert hasattr(transformer, "_mean")
    assert isinstance(transformer.mean, Param)
    assert transformer._mean == 2.0
    assert hasattr(transformer, "column_name")
    assert hasattr(transformer, "_column_name")
    assert isinstance(transformer.column_name, Param)
    assert transformer._column_name == "foo"
    ddf = transformer.transform(input_ddf)
    test_ddf = spark_session.createDataFrame([[0.5], [1.0], [1.5]], "foo: double")
    chispa.assert_df_equality(ddf, test_ddf, ignore_nullable=True)

    path = tmp_path / "transformer_dir"
    path.mkdir()
    p = Pipeline(stages=[transformer])
    p.save(path)
    p = Pipeline.load(path)
    transformer = p.getStages()[0]
    assert isinstance(transformer, SomeTransformerThatWorks)
    assert hasattr(transformer, "__init__")
    assert isinstance(transformer.__init__, types.MethodType)
    assert hasattr(transformer, "setParams")
    assert isinstance(transformer.setParams, types.MethodType)
    assert hasattr(transformer, "_logger")
    assert isinstance(transformer._logger, logging.Logger)
    assert transformer._logger.name == "SomeTransformerThatWorks"
    assert hasattr(transformer, "mean")
    assert hasattr(transformer, "_mean")
    assert isinstance(transformer.mean, Param)
    assert transformer._mean == 2.0
    assert hasattr(transformer, "column_name")
    assert hasattr(transformer, "_column_name")
    assert isinstance(transformer.column_name, Param)
    assert transformer._column_name == "foo"
    ddf = transformer.transform(input_ddf)
    test_ddf = spark_session.createDataFrame([[0.5], [1.0], [1.5]], "foo: double")
    chispa.assert_df_equality(ddf, test_ddf, ignore_nullable=True)


def test_transformerbaseclass_errors():

    # check that it complains about not having an __init__ method
    with pytest.raises(LookupError, match=r".*__init__.*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            ...

    # check that it complains about not having an _transform method
    with pytest.raises(LookupError, match=r".*_transform.*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            def __init__(self):
                ...

    # check that it complains about parameters missing a default value
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            def __init__(self, zz):
                ...

            def _transform(self, ddf):
                ...

    # check that it complains about parameters missing a default value even if another has
    with pytest.raises(SyntaxError, match=r".* zz .*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            def __init__(self, zz, b=None):
                ...

            def _transform(self, ddf):
                ...

    # check that it complains if __init__ method is not calling the parent's __init__ method
    with pytest.raises(SyntaxError, match=r".*super.*.__init()__.*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            def __init__(self, a=None):
                ...

            def _transform(self, ddf):
                ...

    # check that it complains if __init__ method is not using the @pyspark.keyword_only decorator
    with pytest.raises(SyntaxError, match=r".*@pyspark\.keyword_only.*"):

        class SomeTransformer(sbc.TransformerBaseClass):
            def __init__(self, a=None):
                super().__init__()

            def _transform(self, ddf):
                ...
