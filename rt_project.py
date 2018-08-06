from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
import csv

sc =SparkContext()
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/home/ubuntu/final_loan_data.csv')

df.cache()

df = df.select([col(c).cast("double").alias(c) for c in df.columns])
df.printSchema()

(trainingData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
print ("We have %d training examples and %d test examples." % (trainingData.count(), testData.count()))

featureCols = df.columns
featureCols.remove('not_fully_paid')

vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="rawFeatures")

vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", handleInvalid="keep")

rf = RandomForestClassifier(labelCol="not_fully_paid", featuresCol="features", numTrees=10)

pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, rf])

pipelineFit = pipeline.fit(trainingData)
predictions = pipelineFit.transform(testData)

paramGrid = (ParamGridBuilder()\
             .addGrid(rf.maxDepth, [2, 4, 6])\
             .addGrid(rf.maxBins, [20, 60])\
             .addGrid(rf.numTrees, [5, 20])\
             .build())

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="not_fully_paid",metricName="f1")

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cvModel = cv.fit(trainingData)

prediction = cvModel.transform(testData)

selected = prediction.select("not_fully_paid", "prediction", "probability")\
			.orderBy("probability", ascending=False) \
    		.show(n = 10, truncate = 30)

selected.coalesce(1).write.csv("/home/ubuntu/sample.csv")

print("F1: %g" % (evaluator.evaluate(prediction)))


rfModel = pipelineFit.stages[2]

print("Features sorted by their score")
print(sorted(zip(map(lambda x: round(x,4),rfModel.featureImportances), featureCols),reverse=True))