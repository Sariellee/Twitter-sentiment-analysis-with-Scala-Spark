import java.util.Locale

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("SentimentClassifier")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    Locale.getDefault()
    Locale.setDefault(new Locale("en", "US"))
    // Dataset consists of 3 files:

    // imdb_labelled.txt
    val dataset_imdb = spark
      .read
      .format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .load("imdb_labelled.txt")

    // amazon_cells_labelled.txt
    val dataset_amazon = spark
      .read
      .format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .load("amazon_cells_labelled.txt")

    // yelp_labelled.txt
    val dataset_yelp = spark
      .read
      .format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .load("yelp_labelled.txt")

    // merge them together into one DataFrame
    val dataset = dataset_amazon
      .union(dataset_yelp)
      .union(dataset_imdb)
      .toDF("text", "sentiment")

    // random seed for reproducibility
    val randomSeed = 0

    // 80/20 train/test split
    val Array(trainData, testData) = dataset.randomSplit(Array(0.8, 0.2), seed = randomSeed)

    // Divide text into words by regex
    val tokenizer = new RegexTokenizer()
      .setPattern("[^a-zA-Z0-9_!?]")
      .setInputCol("text")
      .setOutputCol("words")

    // remove useless words ("the", "by", "a", "an"...)
    val remover = new StopWordsRemover()
      .setLocale("en_US")
      .setCaseSensitive(true)
      .setInputCol("words")
      .setOutputCol("filteredWords")

    // words into vectors
    val cv = new CountVectorizer()
      .setInputCol("filteredWords")
      .setOutputCol("vectors")
      .setVocabSize(72000)

    // Inverse Document Frequency
    val idf = new IDF()
      .setInputCol("vectors")
      .setOutputCol("features")
      .setMinDocFreq(5)


    val labelIndexer = new StringIndexer()
      .setInputCol("sentiment")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    // chosen model is Support Vector Classifier
    val model = new LogisticRegression().setMaxIter(100)

    // initialize the pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, cv, idf, labelIndexer, model))

    // run it & learn the model
    val pipelineFit = pipeline.fit(trainData)

    // do the predictions on the test set
    val predictions = pipelineFit.transform(testData)

    // get the prediction and label from df
    val results = predictions.select("prediction", "label")

    // results.show(20)
    // predictions.show(20)

    // Instantiate metrics object
    var metrics = new MulticlassMetrics(results.rdd.map(row => {(row.getDouble(0), row.getDouble(1))}))

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val newaccuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $newaccuracy \n")

    // Metrics class for train data
    metrics = new MulticlassMetrics(pipelineFit.transform(trainData)
      .select("prediction", "label").rdd.
      map(row => {(row.getDouble(0), row.getDouble(1))}))

    // Print Fscore for each class
    val labels = metrics.labels
    labels.foreach {l =>
      println(s"F1-Score Train Data($l) = " + metrics.fMeasure(l))
    }

    // show transformations and predictions first 100 test data entries
    predictions.show(100)

    // evaluate the accuracy
    // val evaluator = new BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction")
    // val accuracy = evaluator.evaluate(predictions)
    // println(s"Accuracy = $accuracy")

    // save the model
    pipelineFit.write.overwrite().save("modelLogReg")

    spark.stop()
  }
}