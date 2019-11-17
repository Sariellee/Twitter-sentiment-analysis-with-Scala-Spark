import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

val spark = SparkSession
  .builder
  .appName("SentimentClassifier")
  .config("spark.master", "local")
  .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

Locale.getDefault()
Locale.setDefault(new Locale("en", "US"))

val dataset = spark
  .read
  .format("csv")
  .option("header", "false")
  .option("delimiter", "\t")
  .load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/imdb_labelled.txt")
  .toDF("text", "sentiment")

val model = PipelineModel.load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model")

val randomSeed = 0

val Array(trainData, testData) = dataset.randomSplit(Array(0.8, 0.2), seed = randomSeed)

testData
print(testData.count())
testData.take(5).foreach(println)

val result = model.transform(testData)

