from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
import re
import sys

spark = SparkSession \
        .builder \
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()

spark.sparkContext.setLogLevel("OFF")

inputDF = spark.read.csv(sys.argv[1],header='true', inferSchema='true', sep=';')

# Remove extra quotation marks from csv data
for name in inputDF.schema.names:
    inputDF = inputDF.withColumnRenamed(name, re.sub(r'(^_|_$)','',name.replace('"', '')))
inputDF.printSchema()

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputDF.columns if c != 'quality']
print(featureColumns)
# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns,
                            outputCol="features")

# transform the original data
dataDF_training = assembler.transform(inputDF)


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="quality", outputCol="indexedLabel").fit(dataDF_training)
labelIndexer

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(dataDF_training)

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(dataDF_training)

model_path = 's3://wine-model-643/rfmodel.model'
model.write().overwrite().save(sys.argv[2])


print("Model Successfuly Trained!")
