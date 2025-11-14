import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor, FMRegressor, IsotonicRegression, GeneralizedLinearRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.storage.StorageLevel
import java.io.PrintWriter

// :load ml_prediction/IMDBPredictionModelClean.scala
// IMDBPredictionModelClean.main(Array())

object IMDBPredictionModelClean {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("IMDB Rating Prediction - REAL (Sin Target Encoding)")
      .master("local[*]")
      .config("spark.driver.memory", "10g")
      .config("spark.executor.memory", "10g")
      .config("spark.memory.fraction", "0.8")
      .config("spark.memory.storageFraction", "0.2")
      .config("spark.sql.shuffle.partitions", "100")
      .config("spark.default.parallelism", "100")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    println("=" * 80)
    println("MODELO REAL - SIN TARGET ENCODING - CALIFICACION IMDB")
    println("=" * 80)
    println()
    
    val moviesPath = "./IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb movies.csv"
    val ratingsPath = "./IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb ratings.csv"
    
    println("PASO 1: Cargando y preparando datos...")
    val fullDF = cargarYJoinearDatos(spark, moviesPath, ratingsPath)
    
    println("\nPASO 2: Limpiando datos...")
    val cleanDF = limpiarDatos(fullDF)
    
    println("\nPASO 3: Dividiendo train/test (80/20)...")
    val Array(trainDataRaw, testDataRaw) = cleanDF.randomSplit(Array(0.8, 0.2), seed = 42)
    
    println("\nPASO 4: Aplicando Feature Engineering SIN Target Encoding...")
    val (trainData, testData) = aplicarFeatureEngineering(trainDataRaw, testDataRaw)
    
    val trainOptimized = trainData.repartition(100).persist(StorageLevel.MEMORY_AND_DISK)
    val testOptimized = testData.repartition(100).persist(StorageLevel.MEMORY_AND_DISK)
    
    val trainCount = trainOptimized.count()
    val testCount = testOptimized.count()
    
    println(s"   Datos de entrenamiento: $trainCount filas")
    println(s"   Datos de prueba: $testCount filas")
    
    println("\nPASO 5: Entrenando Ridge Regression...")
    val (baselineModel, baselineMetrics, baselineTime) = entrenarModeloBaseline(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Ridge Regression (Baseline)", baselineMetrics, baselineTime)
    
    println("\nPASO 6: Entrenando Random Forest...")
    val (rfModel, rfMetrics, rfTime) = entrenarRandomForest(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Random Forest", rfMetrics, rfTime)
    imprimirFeatureImportances(rfModel, "Random Forest")
    
    
    println("\nPASO 8: Entrenando Decision Tree...")
    val (dtModel, dtMetrics, dtTime) = entrenarDecisionTree(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Decision Tree", dtMetrics, dtTime)
    imprimirFeatureImportances(dtModel, "Decision Tree")
    
    println("\nPASO 9: Entrenando Lasso Regression...")
    val (lassoModel, lassoMetrics, lassoTime) = entrenarLasso(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Lasso Regression", lassoMetrics, lassoTime)
    
    println("\nPASO 10: Entrenando Elastic Net...")
    val (elasticNetModel, elasticNetMetrics, elasticNetTime) = entrenarElasticNet(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Elastic Net", elasticNetMetrics, elasticNetTime)
    
    println("\nPASO 11: Entrenando Factorization Machines...")
    val (fmModel, fmMetrics, fmTime) = entrenarFM(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Factorization Machines", fmMetrics, fmTime)
    
    println("\nPASO 12: Entrenando Isotonic Regression...")
    val (isotonicModel, isotonicMetrics, isotonicTime) = entrenarIsotonic(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Isotonic Regression", isotonicMetrics, isotonicTime)
    
    println("\nPASO 13: Entrenando Generalized Linear Regression...")
    val (glmModel, glmMetrics, glmTime) = entrenarGLM(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Generalized Linear Regression", glmMetrics, glmTime)
    
    println("\nPASO 14: Generando reporte comparativo...")
    generarReporteComparativo(
      Map(
        "Ridge Regression" -> (baselineMetrics, baselineTime),
        "Random Forest" -> (rfMetrics, rfTime),
        "Decision Tree" -> (dtMetrics, dtTime),
        "Lasso Regression" -> (lassoMetrics, lassoTime),
        "Elastic Net" -> (elasticNetMetrics, elasticNetTime),
        "Factorization Machines" -> (fmMetrics, fmTime),
        "Isotonic Regression" -> (isotonicMetrics, isotonicTime),
        "Generalized Linear Regression" -> (glmMetrics, glmTime)
      )
    )
    
    trainOptimized.unpersist()
    testOptimized.unpersist()
    
    println("\n" + "=" * 80)
    println("PROCESO COMPLETADO")
    println("=" * 80)
    
    spark.stop()
  }
  
  def cargarYJoinearDatos(spark: SparkSession, moviesPath: String, ratingsPath: String): DataFrame = {
    val movies = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("escape", "\"")
      .option("multiLine", "true")
      .csv(moviesPath)
    
    val ratings = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(ratingsPath)
    
    val joinedDF = movies.join(ratings, Seq("imdb_title_id"), "inner")
    println(s"   Dataset completo: ${joinedDF.count()} filas")
    
    joinedDF
  }
  
  def limpiarDatos(df: DataFrame): DataFrame = {
    val originalCount = df.count()
    
    val dfYearCleaned = df.withColumn(
      "year_clean",
      regexp_replace(col("year"), "[^0-9]", "").cast(IntegerType)
    )
    
    val cleanDF = dfYearCleaned.na.drop(Seq(
      "description", "genre", "director", "actors",
      "duration", "avg_vote", "year_clean"
    ))
    
    val cleanCount = cleanDF.count()
    val lossPercent = ((originalCount - cleanCount).toDouble / originalCount * 100)
    
    println(s"   Filas originales: $originalCount")
    println(s"   Filas limpias: $cleanCount")
    println(f"   Perdida: $lossPercent%.2f%%")
    
    cleanDF
  }
  
  def aplicarFeatureEngineering(
    trainRaw: DataFrame, 
    testRaw: DataFrame
  ): (DataFrame, DataFrame) = {
    
    println("   Aplicando Frequency Encoding para director...")
    val (trainWithDirector, directorFreqMap) = frequencyEncodeOnTrain(trainRaw, "director")
    val testWithDirector = applyFrequencyEncoding(testRaw, "director", directorFreqMap)
    
    println("   Aplicando Frequency Encoding para actors...")
    val (trainWithActors, actorsFreqMap) = frequencyEncodeOnTrain(trainWithDirector, "actors")
    val testWithActors = applyFrequencyEncoding(testWithDirector, "actors", actorsFreqMap)
    
    println("   Reduciendo cardinalidad de genre...")
    val (trainWithGenre, genreTopCategories) = reduceCardinalityOnTrain(trainWithActors, "genre", topN = 30)
    val testWithGenre = applyCardinalityReduction(testWithActors, "genre", genreTopCategories)
    
    println("   Creando features derivadas...")
    val trainEnriched = crearFeaturesDerivadasSeguras(trainWithGenre)
    val testEnriched = crearFeaturesDerivadasSeguras(testWithGenre)
    
    println("   Feature Engineering completado")
    
    (trainEnriched, testEnriched)
  }
  
  // FREQUENCY ENCODING (NO usa el target)
  def frequencyEncodeOnTrain(
    df: DataFrame, 
    column: String
  ): (DataFrame, Map[String, Double]) = {
    
    val totalCount = df.count().toDouble
    
    val freqByCategory = df.groupBy(column)
      .agg(count("*").alias("category_count"))
      .collect()
    
    val frequencyMap = freqByCategory.map { row =>
      val category = row.getString(0)
      val categoryCount = row.getLong(1)
      val frequency = categoryCount / totalCount
      category -> frequency
    }.toMap
    
    val defaultFrequency = 1.0 / totalCount
    val frequencyMapWithDefault = frequencyMap + ("__UNKNOWN__" -> defaultFrequency)
    
    val dfEncoded = applyFrequencyEncoding(df, column, frequencyMapWithDefault)
    
    (dfEncoded, frequencyMapWithDefault)
  }
  
  def applyFrequencyEncoding(
    df: DataFrame, 
    column: String, 
    frequencyMap: Map[String, Double]
  ): DataFrame = {
    
    val defaultValue = frequencyMap.getOrElse("__UNKNOWN__", 0.0001)
    
    val encodeUDF = udf((value: String) => 
      frequencyMap.getOrElse(value, defaultValue)
    )
    
    df.withColumn(s"${column}_freq", encodeUDF(col(column)))
      .drop(column)
  }
  
  def reduceCardinalityOnTrain(
    df: DataFrame, 
    column: String, 
    topN: Int
  ): (DataFrame, Set[String]) = {
    
    val topCategories = df.groupBy(column)
      .count()
      .orderBy(desc("count"))
      .limit(topN)
      .select(column)
      .collect()
      .map(_.getString(0))
      .toSet
    
    val dfReduced = applyCardinalityReduction(df, column, topCategories)
    
    (dfReduced, topCategories)
  }
  
  def applyCardinalityReduction(
    df: DataFrame, 
    column: String, 
    topCategories: Set[String]
  ): DataFrame = {
    
    val categorizeUDF = udf((value: String) => 
      if (topCategories.contains(value)) value else "Other"
    )
    
    df.withColumn(column, categorizeUDF(col(column)))
  }
  
  def crearFeaturesDerivadasSeguras(df: DataFrame): DataFrame = {
    df
      .withColumn("decade", 
        (col("year_clean") / 10).cast(IntegerType) * 10)
      .withColumn("is_recent", 
        when(col("year_clean") >= 2015, 1.0).otherwise(0.0))
      .withColumn("is_old_classic", 
        when(col("year_clean") <= 1980, 1.0).otherwise(0.0))
      .withColumn("duration_category",
        when(col("duration") <= 90, "short")
        .when(col("duration") <= 120, "medium")
        .otherwise("long"))
  }
  
  def crearPipeline(regressor: org.apache.spark.ml.Estimator[_]): Pipeline = {
    val descTokenizer = new RegexTokenizer()
      .setInputCol("description")
      .setOutputCol("desc_words")
      .setPattern("\\W+")
      .setMinTokenLength(3)
    
    val descRemover = new StopWordsRemover()
      .setInputCol("desc_words")
      .setOutputCol("desc_filtered")
    
    val descHashingTF = new HashingTF()
      .setInputCol("desc_filtered")
      .setOutputCol("desc_tf")
      .setNumFeatures(100)
    
    val descIDF = new IDF()
      .setInputCol("desc_tf")
      .setOutputCol("description_features")
    
    val genreHasher = new FeatureHasher()
      .setInputCols(Array("genre"))
      .setOutputCol("genre_features")
      .setNumFeatures(16)
    
    val durationIndexer = new StringIndexer()
      .setInputCol("duration_category")
      .setOutputCol("duration_indexed")
      .setHandleInvalid("keep")
    
    // Assembler - USA FREQUENCY ENCODING (no target encoding)
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "description_features",    // 100 features
        "genre_features",           // 16 features
        "director_freq",            // 1 feature (FREQUENCY, no target)
        "actors_freq",              // 1 feature (FREQUENCY, no target)
        "duration",                 // 1 feature
        "duration_indexed",         // 1 feature
        "year_clean",               // 1 feature
        "decade",                   // 1 feature
        "is_recent",                // 1 feature
        "is_old_classic"            // 1 feature
      ))
      .setOutputCol("features")
      .setHandleInvalid("skip")
    
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
      .setWithStd(true)
      .setWithMean(false)
    
    new Pipeline().setStages(Array(
      descTokenizer, descRemover,
      descHashingTF, descIDF,
      genreHasher,
      durationIndexer,
      assembler,
      scaler,
      regressor
    ))
  }
  
  def entrenarModeloBaseline(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val lr = new LinearRegression()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .setTol(1e-6)
    
    val pipeline = crearPipeline(lr)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarRandomForest(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val rf = new RandomForestRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setNumTrees(30)
      .setMaxDepth(8)
      .setMinInstancesPerNode(10)
      .setSubsamplingRate(0.8)
      .setSeed(42)
    
    val pipeline = crearPipeline(rf)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarGBT(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val gbt = new GBTRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxIter(50)
      .setMaxDepth(5)
      .setStepSize(0.1)
      .setSubsamplingRate(0.8)
      .setSeed(42)
    
    val pipeline = crearPipeline(gbt)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarDecisionTree(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val dt = new DecisionTreeRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxDepth(10)
      .setMinInstancesPerNode(20)
      .setMinInfoGain(0.0)
      .setSeed(42)
    
    val pipeline = crearPipeline(dt)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarLasso(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val lasso = new LinearRegression()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(1.0)
      .setTol(1e-6)
    
    val pipeline = crearPipeline(lasso)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarElasticNet(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val elasticNet = new LinearRegression()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)
      .setTol(1e-6)
    
    val pipeline = crearPipeline(elasticNet)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarFM(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val fm = new FMRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setFactorSize(8)
      .setFitIntercept(true)
      .setFitLinear(true)
      .setRegParam(0.01)
      .setMiniBatchFraction(0.8)
      .setInitStd(0.01)
      .setMaxIter(100)
      .setStepSize(0.001)
      .setSeed(42)
    
    val pipeline = crearPipeline(fm)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarIsotonic(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val isotonic = new IsotonicRegression()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setIsotonic(true)
    
    val pipeline = crearPipeline(isotonic)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def entrenarGLM(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    val startTime = System.nanoTime()
    
    val glm = new GeneralizedLinearRegression()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setFamily("gaussian")
      .setLink("identity")
      .setMaxIter(100)
      .setRegParam(0.1)
      .setTol(1e-6)
    
    val pipeline = crearPipeline(glm)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def evaluarModelo(predictions: DataFrame): Map[String, Double] = {
    val evaluator = new RegressionEvaluator()
      .setLabelCol("avg_vote")
      .setPredictionCol("prediction")
    
    val rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    val mae = evaluator.setMetricName("mae").evaluate(predictions)
    val r2 = evaluator.setMetricName("r2").evaluate(predictions)
    val mse = evaluator.setMetricName("mse").evaluate(predictions)
    
    Map("RMSE" -> rmse, "MAE" -> mae, "R2" -> r2, "MSE" -> mse)
  }
  
  def imprimirMetricas(modelName: String, metrics: Map[String, Double], time: Double): Unit = {
    println(s"\n   Metricas de $modelName:")
    println(f"      RMSE: ${metrics("RMSE")}%.4f")
    println(f"      MAE:  ${metrics("MAE")}%.4f")
    println(f"      R²:   ${metrics("R2")}%.4f")
    println(f"      MSE:  ${metrics("MSE")}%.4f")
    if (time > 0) {
      println(f"      Tiempo: ${time}%.2f segundos (${time/60}%.2f minutos)")
    }
  }
  
  def imprimirFeatureImportances(model: PipelineModel, modelName: String): Unit = {
    val treeModel = model.stages.last match {
      case rf: org.apache.spark.ml.regression.RandomForestRegressionModel => 
        Some(rf.featureImportances)
      case gbt: org.apache.spark.ml.regression.GBTRegressionModel => 
        Some(gbt.featureImportances)
      case dt: org.apache.spark.ml.regression.DecisionTreeRegressionModel => 
        Some(dt.featureImportances)
      case _ => None
    }
    
    treeModel.foreach { importances =>
      println(s"\n   Feature Importances (Top 10):")
      val topFeatures = importances.toArray.zipWithIndex
        .sortBy(-_._1)
        .take(10)
      
      topFeatures.foreach { case (importance, idx) =>
        val featureName = idx match {
          case i if i >= 0 && i < 100 => s"description[$i]"
          case i if i >= 100 && i < 116 => s"genre[${i-100}]"
          case 116 => "director_freq"
          case 117 => "actors_freq"
          case 118 => "duration"
          case 119 => "duration_indexed"
          case 120 => "year_clean"
          case 121 => "decade"
          case 122 => "is_recent"
          case 123 => "is_old_classic"
          case _ => s"unknown[$idx]"
        }
        println(f"      $featureName%-25s: ${importance * 100}%.2f%%")
      }
    }
  }
  
  def generarReporteComparativo(modelos: Map[String, (Map[String, Double], Double)]): Unit = {
    val outputPath = "ml-prediction/resultados/reporte_clean.txt"
    val writer = new PrintWriter(outputPath)
    
    writer.println("=" * 80)
    writer.println("REPORTE - MODELO REAL (SIN TARGET ENCODING)")
    writer.println("=" * 80)
    writer.println()
    
    writer.println("CAMBIOS RESPECTO AL MODELO ANTERIOR:")
    writer.println("-" * 80)
    writer.println("ELIMINADO: Target Encoding (director_encoded, actors_encoded)")
    writer.println("AGREGADO:  Frequency Encoding (director_freq, actors_freq)")
    writer.println()
    writer.println("RAZÓN DEL CAMBIO:")
    writer.println("  El modelo anterior tenía R² = 0.86 con:")
    writer.println("  - actors_encoded: 74% de feature importance")
    writer.println("  - director_encoded: 11% de feature importance")
    writer.println("  → Target Encoding creaba correlación casi perfecta con el target")
    writer.println("  → Era data leakage indirecto (usaba promedio del target)")
    writer.println()
    
    writer.println("FEATURES UTILIZADAS:")
    writer.println("-" * 80)
    writer.println("  • Description: TF-IDF (100 features)")
    writer.println("  • Genre: Feature Hashing (16 features)")
    writer.println("  • Director: Frequency Encoding (1 feature) SIN target")
    writer.println("  • Actors: Frequency Encoding (1 feature) SIN target")
    writer.println("  • Duration: Numérica + Categórica (2 features)")
    writer.println("  • Year: Numérica + Derivadas (4 features)")
    writer.println("  • TOTAL: ~124 features")
    writer.println()
    
    writer.println("RESULTADOS:")
    writer.println("=" * 80)
    writer.println(f"${"Modelo"}%-35s ${"RMSE"}%-10s ${"MAE"}%-10s ${"R²"}%-10s ${"Tiempo"}%-15s")
    writer.println("-" * 80)
    
    modelos.toSeq.sortBy(_._2._1("RMSE")).foreach { case (nombre, (metricas, tiempo)) =>
      val tiempoStr = if (tiempo > 0) f"${tiempo/60}%.2f min" else "N/A"
      writer.println(
        f"$nombre%-35s ${metricas("RMSE")}%-10.4f ${metricas("MAE")}%-10.4f ${metricas("R2")}%-10.4f $tiempoStr%-15s"
      )
    }
    
    writer.println("=" * 80)
    writer.println()
    
    val mejorModelo = modelos.minBy(_._2._1("RMSE"))
    
    writer.println("INTERPRETACION:")
    writer.println("-" * 80)
    writer.println(s"Mejor modelo: ${mejorModelo._1}")
    writer.println(f"   - R²: ${mejorModelo._2._1("R2")}%.4f")
    writer.println()
    
    val r2 = mejorModelo._2._1("R2")
    if (r2 < 0.4) {
      writer.println("R² REALISTA (< 0.4)")
      writer.println("   Predice ratings usando SOLO caracteristicas intrinsecas")
      writer.println("   NO hay data leakage de ningun tipo")
      writer.println("   Este es el verdadero poder predictivo del modelo")
    } else if (r2 < 0.6) {
      writer.println("R² ACEPTABLE (0.4-0.6)")
      writer.println("   Buena capacidad predictiva sin data leakage")
    } else {
      writer.println("R² ALTO (> 0.6)")
      writer.println("   Verificar posible leakage residual")
    }
    
    writer.println()
    writer.println("=" * 80)
    writer.close()
    
    println(s"   Reporte guardado en: $outputPath")
    
    println("\n" + "=" * 80)
    println("RESUMEN FINAL (MODELO REAL)")
    println("=" * 80)
    println(f"${"Modelo"}%-35s ${"RMSE"}%-10s ${"R²"}%-10s")
    println("-" * 80)
    modelos.toSeq.sortBy(_._2._1("RMSE")).foreach { case (nombre, (metricas, _)) =>
      println(f"$nombre%-35s ${metricas("RMSE")}%-10.4f ${metricas("R2")}%-10.4f")
    }
    println("=" * 80)
  }
}
