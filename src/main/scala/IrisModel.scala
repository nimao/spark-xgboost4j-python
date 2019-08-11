import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModelBridge
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import ml.dmlc.xgboost4j.scala.XGBoost

package ml.dmlc.xgboost4j.scala.spark {
  import ml.dmlc.xgboost4j.scala.Booster
  class XGBoostClassificationModelBridge(uid: String, numClasses: Int,_booster: Booster) {
      val xgbClassificationModel = new XGBoostClassificationModel(uid, numClasses, _booster)
  }
}

object IrisModel {
  def main(args: Array[String]) {
  val spark = SparkSession.builder.appName("Iris Model").getOrCreate()
  val schema = new StructType(Array(
    StructField("sepal length", DoubleType, true),
    StructField("sepal width", DoubleType, true),
    StructField("petal length", DoubleType, true),
    StructField("petal width", DoubleType, true),
    StructField("class", StringType, true)))
  val rawInput = spark.read.schema(schema).csv("iris.data")

  val stringIndexer = new StringIndexer().
    setInputCol("class").
    setOutputCol("classIndex").
    fit(rawInput)

  val labelTransformed = stringIndexer.transform(rawInput).drop("class")
  val vectorAssembler = new VectorAssembler().
    setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
    setOutputCol("features")
  val xgbInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")

  val xgbParam = Map("eta" -> 0.1f,
    "max_depth" -> 2,
    "objective" -> "multi:softprob",
    "num_class" -> 3,
    "num_round" -> 100,
    "num_workers" -> 2)
  val xgbClassifier = new XGBoostClassifier(xgbParam).
    setFeaturesCol("features").
    setLabelCol("classIndex")

// val xgbClassificationModel = xgbClassifier.fit(xgbInput)

  val booster = XGBoost.loadModel("python_xgb.model")
  val xgbClassificationModel = new XGBoostClassificationModelBridge("python_xgb", 3, booster)

// val features = xgbInput.head().getAs[Vector]("features")
  val results = xgbClassificationModel.xgbClassificationModel.transform(xgbInput)

  results.collect.foreach(println)

  spark.stop()
  }
}
