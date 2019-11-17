import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object stream_read {
  def main(args: Array[String]): Unit = {

    val spark = new sql.SparkSession.Builder().master("local[*]").appName("Twitter-Sentiment").getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")
    val ssc = new StreamingContext(spark.sparkContext, Seconds(4))


    Locale.getDefault()
    Locale.setDefault(new Locale("en", "US"))
    val model = PipelineModel.load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model")
    val outputPath = "/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/output_model"


    val readStream = ssc.socketTextStream("localhost", 9999)
    readStream.foreachRDD(
      (rdd, time) =>
        if (!rdd.isEmpty()) {
          val df = rdd.toDF("text")
          // do something
          model
            .transform(df)
            .withColumn("timestamp", current_timestamp())
            .select("timestamp", "text", "prediction")
            .write
//            .mode(SaveMode.Append)
            .csv(outputPath)
          if (df.count() > 0) {
            model
              .transform(df)
              .withColumn("timestamp", current_timestamp())
              .select("timestamp", "text", "prediction")
              .write
//              .mode(SaveMode.Append)
              .csv(outputPath)
          }
          model.transform(df).collect().foreach(println)
        }
    )

  }
}

//val lines = ssc.socketTextStream("localhost", 8989)
//lines.foreachRDD {
//  rdd =>
//    val df = rdd.toDF().withColumnRenamed("value", "SentimentText")
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