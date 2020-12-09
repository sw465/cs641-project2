from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import re
import sys

print("sys args are : ")
print(sys.argv)

model_path = ('s3n://wine-model-643/rfmodel.model/')
test_data_path = ('s3n://training-dataset-643/ValidationDataset.csv')

spark = SparkSession \
        .builder \
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()

spark.sparkContext.setLogLevel("OFF")
inputDF_validation = spark.read.csv(sys.argv[1],header='true', inferSchema='true', sep=';')

for name in inputDF_validation.schema.names:
    inputDF_validation = inputDF_validation.withColumnRenamed(name, re.sub(r'(^_|_$)','',name.replace('"', '')))

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputDF_validation.columns if c != 'quality']

# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns,
                            outputCol="features")

# transform the original data
dataDF_validation = assembler.transform(inputDF_validation)

model_unpacked = PipelineModel.load('rfmodel.model')

predictions = model_unpacked.transform(dataDF_validation)

evaluator = MulticlassClassificationEvaluator(
labelCol="indexedLabel", predictionCol="prediction", metricName="f1")

f1_score = evaluator.evaluate(predictions)
print("f1 score = " + str(f1_score))
