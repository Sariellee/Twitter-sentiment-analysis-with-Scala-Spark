import java.io.File
import java.util.Locale

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.streaming._
import org.apache.spark.sql.functions._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object stream_read {
  def main(args: Array[String]): Unit = {

    val spark = new sql.SparkSession.Builder().master("local[1]").appName("Twitter-Sentiment").getOrCreate()
    import spark.implicits._
    val ssc = new StreamingContext(spark.sparkContext, Seconds(2))
    ssc.checkpoint(new File("model", "streaming_checkpoint").toString)

    Locale.getDefault()
    Locale.setDefault(new Locale("en", "US"))
    val model = PipelineModel.load("model")
    val outputPath = "output_model"


    val readStream = ssc.socketTextStream("", 8989)
    readStream.foreachRDD(
      rdd =>
        if (!rdd.isEmpty()) {
          rdd.foreach(println)
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
