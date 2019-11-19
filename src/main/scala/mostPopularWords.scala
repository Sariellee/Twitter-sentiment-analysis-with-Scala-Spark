import org.apache.spark.ml.feature.StopWordsRemover
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


    val remover = new StopWordsRemover()
      .setLocale("en_US")
      .setCaseSensitive(true)
      .setInputCol("words")
      .setOutputCol("filteredWords")

    val dataset = sc.textFile("output_model")

    val wordsToRemove = remover.getStopWords
    val regex = "\\b(" + wordsToRemove.mkString("|") + ")\\W"

    val counts = dataset
      .flatMap(line => line
        .substring(line.indexOf(",") + 1) // remove timestamp
        .replaceAll("""[^a-zA-Z2-9_ ]+""", "") // remove all symbols which are not letters or numbers
        .replaceAll(regex, "") // remove all the stop words
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
