import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import substring, col
from warnings import filterwarnings
import dataframe_image as dfi
filterwarnings('ignore')

def twin():
    spark = SparkSession.builder.master("local").appName("wind_turbine_project").getOrCreate()
    sc = spark.sparkContext
    spark_df = spark.read.csv('T1.csv', header=True, inferSchema=True)
    spark_df.cache()
    spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
    spark_df = spark_df.withColumn("month", substring("date/time", 4,2))
    spark_df = spark_df.withColumn("hour", substring("date/time", 12,2))
    spark_df = spark_df.withColumn("day", substring("date/time", 12,2))
    spark_df = spark_df.withColumn('month', spark_df.month.cast(IntegerType()))
    spark_df = spark_df.withColumn('hour', spark_df.hour.cast(IntegerType()))
    spark_df = spark_df.withColumn('day', spark_df.month.cast(IntegerType()))
    pd.options.display.float_format = '{:.2f}'.format
    spark_df.select('wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)').toPandas().describe()
    sample_df = spark_df.sample(withReplacement=False, fraction=0.1, seed=42).toPandas()
    columns = ['wind speed (m/s)', 'wind direction (°)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']
    wind_speed = spark_df.select('wind speed (m/s)').toPandas()
    spark_df = spark_df.withColumn('label', spark_df['lv activepower (kw)'])
    variables = ['month', 'hour', 'wind speed (m/s)', 'wind direction (°)']
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    va_df = vectorAssembler.transform(spark_df)
    final_df = va_df.select('features', 'label')
    splits = final_df.randomSplit([0.8, 0.2])
    train_df = splits[0]
    test_df = splits[1]
    print('Train dataset: ', train_df.count())
    print('Test dataset : ', test_df.count())
    gbm = GBTRegressor(featuresCol='features', labelCol='label')
    gbm_model = gbm.fit(train_df)
    y_pred = gbm_model.transform(test_df)
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
    eva_df = spark.createDataFrame(sample_df)
    eva_df = eva_df.withColumn('label', eva_df['lv activepower (kw)'])
    variables = ['month', 'hour', 'wind speed (m/s)', 'wind direction (°)']
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    vec_df = vectorAssembler.transform(eva_df)
    vec_df = vec_df.select('features', 'label')
    preds = gbm_model.transform(vec_df)
    preds_df = preds.select('label','prediction').toPandas()
    frames = [sample_df[['wind speed (m/s)', 'theoretical_power_curve (kwh)']], preds_df]
    sample_data = pd.concat(frames, axis=1)
    return sample_data

if __name__ == '__main__':
    print(twin())