from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
import time
spark = SparkSession.builder \
    .appName("Lab6") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)
df = spark.read.csv("adult.csv", header=False, inferSchema=True)
cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"]
for i,c in enumerate(cols):
    df = df.withColumnRenamed(f"_c{i}", c)
print(f"Total records: {df.count()}")
print("\nIncome column unique values:")
df.select("income").distinct().show(10)
print("\nIncome distribution:")
df.groupBy("income").count().show()
print("=" * 60)
print("STEP 2: Creating Label Column")
print("=" * 60)
df = df.withColumn("label", when(col("income").contains(">50K"), 1.0).otherwise(0.0))
df = df.drop("income")
print("\nLabel distribution:")
df.groupBy("label").count().show()
label_counts = df.groupBy("label").count().collect()
if len(label_counts) == 1:
    print("\nERROR: Only one label class found!")
    print("Checking raw data...")
    df_raw = spark.read.csv("adult.csv", header=False)
    df_raw.select("_c14").distinct().show(20)
    spark.stop()
    exit()
print("=" * 60)
print("STEP 3: Building ML Pipeline")
print("=" * 60)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
categorical = ["workclass","education","marital_status","occupation",
               "relationship","race","sex","native_country"]
numerical = ["age","fnlwgt","education_num","capital_gain",
             "capital_loss","hours_per_week"]
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") 
            for c in categorical]
assembler = VectorAssembler(
    inputCols=[c+"_idx" for c in categorical] + numerical,
    outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", 
                        fitIntercept=True, maxIter=10)
pipeline = Pipeline(stages=indexers + [assembler, lr])
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train.count()}, Test size: {test.count()}")
print("\nTraining Logistic Regression...")
t0 = time.time()
model = pipeline.fit(train)
print(f"Training time: {time.time()-t0:.3f}s")
predictions = model.transform(test)
print("\nPrediction distribution:")
predictions.groupBy("label", "prediction").count().show()
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"\nAUC-ROC: {auc:.4f}")
print("=" * 60)
print("STEP 4: Partition Scaling")
print("=" * 60)
for parts in [1,2,4,8,16]:
    t0 = time.time()
    pipeline.fit(train.repartition(parts))
    print(f"Partitions: {parts:2} - Time: {time.time()-t0:.3f}s")
print("=" * 60)
print("STEP 5: Broadcast vs Shuffle Join")
print("=" * 60)
from pyspark.sql.functions import broadcast
lookup = spark.createDataFrame([(i, f"cat_{i}") for i in range(100)], ["id", "category"])
large = train.select((col("age") % 100).alias("id"), "label")
t0 = time.time()
large.join(lookup, "id").count()
t1 = time.time()
print(f"Without broadcast: {t1-t0:.3f}s")
t0 = time.time()
large.join(broadcast(lookup), "id").count()
t1 = time.time()
print(f"With broadcast: {t1-t0:.3f}s")
spark.stop()
print("\nLab 6 Complete!")
