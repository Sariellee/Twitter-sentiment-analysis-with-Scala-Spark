import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.streaming.{Seconds, StreamingContext}

//val spark = SparkSession
//  .builder
//  .appName("SentimentClassifier")
//  .config("spark.master", "local")
//  .getOrCreate()
//spark.sparkContext.setLogLevel("ERROR")


val spark = new sql.SparkSession.Builder().master("local[*]").appName("Twitter-Sentiment").getOrCreate()
import spark.implicits._
spark.sparkContext.setLogLevel("ERROR")

val ssc = new StreamingContext(spark.sparkContext, Seconds(60))


Locale.getDefault()
Locale.setDefault(new Locale("en", "US"))
val model = PipelineModel.load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model")



val readFromKafka = ssc.socketTextStream("localhost", 9999)
readFromKafka.foreachRDD(
  (rdd, time) =>
    if (!rdd.isEmpty()) {
      val df = rdd.toDF
      // do something
      model.transform(df).collect().foreach(println)
    }
)

//val lines = ssc.socketTextStream("localhost", 8989)
//lines.foreachRDD {
//  rdd =>
//    val df = rdd.toDF().withColumnRenamed("еуче", "SentimentText")
//    if (df.count() > 0) {
//      pipelineFit
//        .transform(df)
//        .withColumn("timestamp", current_timestamp())
//        .select("timestamp", "SentimentText", "prediction")
//        .repartition(1)
//        .coalesce(1)
//        .write
//        .mode(SaveMode.Append)
//        .csv(outputPath)
//    }
//}





ssc.start()
ssc.awaitTermination()
