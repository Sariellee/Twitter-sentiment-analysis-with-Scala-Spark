import java.io.File
import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object STREAM_LISTENER {
  def main(args: Array[String]): Unit = {

    val spark = new sql.SparkSession.Builder().master("local[*]").appName("Twitter-Sentiment").getOrCreate()
    import spark.implicits._
    val ssc = new StreamingContext(spark.sparkContext, Seconds(10))
    ssc.checkpoint(new File("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model", "streaming_checkpoint").toString)

    Locale.getDefault()
    Locale.setDefault(new Locale("en", "US"))
    val model = PipelineModel.load("/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/model")
    val outputPath = "/Users/semenkiselev/IdeaProjects/Twitter-sentiment-analysis-with-Scala-Spark/output_model"


    val readStream = ssc.socketTextStream("localhost", 9999)
    readStream.foreachRDD(
      (rdd, time) =>
        if (!rdd.isEmpty()) {
          rdd.foreach(println)
          print(rdd)
          val df = rdd.toDF("text")
          if (df.count() > 0) {
            model
              .transform(df)
              .withColumn("timestamp", current_timestamp())
              .select("timestamp", "text", "prediction")
              .write
              .mode(SaveMode.Append)
              .csv(outputPath)
          }
          model.transform(df).collect().foreach(println)
        }
    )
    ssc.start()
    ssc.awaitTermination()
  }
}

