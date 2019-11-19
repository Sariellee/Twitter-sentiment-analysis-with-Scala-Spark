import org.apache.spark.sql.SparkSession


object mostPopularWords {
    def main(args: Array[String]): Unit = {
      val spark = SparkSession
        .builder
        .appName("SentimentClassifier")
        .config("spark.master", "local")
        .getOrCreate()
      spark.sparkContext.setLogLevel("ERROR")
      val sc = spark.sparkContext

      val dataset = sc.textFile("output_model")

      val counts = dataset
        .flatMap(line => line
          .substring(line.indexOf(",")+1)
          .replaceAll("""[^a-zA-Z2-9_ ]+""", "")
          .split(" ")).
        map(word => (word, 1)).
        reduceByKey(_ + _)

      val mostCommon = counts
        .map(p => (p._2, p._1))
        .sortByKey(false, 1)

      val tenCommon = mostCommon.take(10)
      sc.parallelize(tenCommon).saveAsTextFile("mostPopularWords.txt")
    }
}
