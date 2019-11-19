import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession


object Word2VecNN {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("SentimentClassifier")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val dataset = spark
      .read
      .format("csv")
      .option("header", "true")
      .load("train.csv")
      .withColumnRenamed("Sentiment", "target")
      .drop("ItemId")

    val tokenizer = new RegexTokenizer()
      .setInputCol("SentimentText")
      .setOutputCol("words")
      .setPattern("\\W")

    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("features")
      .setVectorSize(100)
      .setMinCount(0)

    val labelIndexer = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("label")

    val Array(trainData, testData) = dataset.randomSplit(Array(0.8, 0.2))

    val layers = Array[Int](100, 75, 10, 2)
    val classifier = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)


    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, word2Vec, labelIndexer, classifier))

    val pipelineFit = pipeline.fit(trainData)

    val predictions = pipelineFit.transform(testData)

    predictions
      .select("label", "prediction")
      .coalesce(1)
      .write
      .csv("predictions.csv")

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("rawPrediction")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy = $accuracy")

    spark.stop()
  }
}
