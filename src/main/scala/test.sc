import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

val spark = SparkSession
  .builder
  .appName("SentimentClassifier")
  .config("spark.master", "local")
  .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

val someData = Seq(
  ("t1", 0),
  ("t12", 0)
)


Locale.getDefault()
Locale.setDefault(new Locale("en", "US"))
val model = PipelineModel.load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model")


val df = spark.createDataFrame(someData).toDF(colNames = "text", "sentiment")
print(df)
df.collect().foreach(println)

val result = model.transform(df)

result.collect().foreach(println)
result.show(2)

