import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.current_timestamp
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object TwitterProcessing {
  def main(args: Array[String]): Unit = {
    val outputPath = "output"
    val spark = SparkSession
      .builder
      .appName("TwitterProcessing")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val pipelineFit = PipelineModel.load("model")
    import spark.implicits._

    val ssc = new StreamingContext(spark.sparkContext, Seconds(2))
    val lines = ssc.socketTextStream("localhost", 8989)
    lines.foreachRDD {
      rdd =>
        val df = rdd.toDF().withColumnRenamed("sentiment", "text")
        if (df.count() > 0) {
          pipelineFit
            .transform(df)
            .withColumn("timestamp", current_timestamp())
            .select("timestamp", "text", "prediction")
            .repartition(1)
            .coalesce(1)
            .write
            .mode(SaveMode.Append)
            .csv(outputPath)
        }
    }

    ssc.start()
    ssc.awaitTermination()
  }
}
