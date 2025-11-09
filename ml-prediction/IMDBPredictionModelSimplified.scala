import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.storage.StorageLevel
import java.io.PrintWriter

// :load ml_prediction/IMDBPredictionModelSimplified.scala
// IMDBPredictionModelSimplified.main(Array())

object IMDBPredictionModelSimplified {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("IMDB Rating Prediction - Simplified")
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
    println("MODELO DE PREDICCIÓN SIMPLIFICADO - CALIFICACIÓN IMDB")
    println("=" * 80)
    println()
    
    val moviesPath = "IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb movies.csv"
    val ratingsPath = "IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb ratings.csv"
    
    println("📊 PASO 1: Cargando y preparando datos...")
    val fullDF = cargarYJoinearDatos(spark, moviesPath, ratingsPath)
    
    println("\n🧹 PASO 2: Limpiando datos...")
    val cleanDF = limpiarYEnriquecerDatos(fullDF)
    
    println("\n🔧 PASO 3: Dividiendo train/test (80/20) ANTES de Feature Engineering...")
    println("   ⚠️  CRÍTICO: Split ANTES de Target Encoding para evitar data leakage")
    val Array(trainDataRaw, testDataRaw) = cleanDF.randomSplit(Array(0.8, 0.2), seed = 42)
    
    println("\n🎯 PASO 4: Aplicando Feature Engineering SOLO en train...")
    val (trainData, testData, encodingMaps) = aplicarFeatureEngineeringConSplit(trainDataRaw, testDataRaw)
    
    val trainOptimized = trainData.repartition(100).persist(StorageLevel.MEMORY_AND_DISK)
    val testOptimized = testData.repartition(100).persist(StorageLevel.MEMORY_AND_DISK)
    
    val trainCount = trainOptimized.count()
    val testCount = testOptimized.count()
    
    println(s"   ✓ Datos de entrenamiento: $trainCount filas")
    println(s"   ✓ Datos de prueba: $testCount filas")
    println(s"   ✓ Encodings calculados solo en train (sin data leakage)")
    println(s"   ✓ Director encodings: ${encodingMaps("director").size} categorías")
    println(s"   ✓ Actors encodings: ${encodingMaps("actors").size} categorías")
    
    // ============================================================================
    // MODELO 1: Ridge Regression (Baseline)
    // ============================================================================
    println("\n📈 PASO 5: Entrenando BASELINE (Ridge Regression)...")
    val (baselineModel, baselineMetrics, baselineTime) = entrenarModeloBaseline(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Ridge Regression (Baseline)", baselineMetrics, baselineTime)
    guardarPredicciones(
      baselineModel, 
      testOptimized, 
      "ml_prediction/resultados/simplified_baseline_predictions.txt"
    )
    
    // ============================================================================
    // MODELO 2: Random Forest (Hiperparámetros fijos optimizados)
    // ============================================================================
    println("\n🌲 PASO 6: Entrenando Random Forest (hiperparámetros optimizados)...")
    val (rfModel, rfMetrics, rfTime) = entrenarRandomForest(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Random Forest", rfMetrics, rfTime)
    imprimirFeatureImportances(rfModel, "Random Forest")
    guardarPredicciones(
      rfModel, 
      testOptimized, 
      "ml_prediction/resultados/simplified_rf_predictions.txt"
    )
    
    // ============================================================================
    // MODELO 3: Gradient Boosted Trees (Hiperparámetros fijos optimizados)
    // ============================================================================
    println("\n🚀 PASO 7: Entrenando GBT (hiperparámetros optimizados)...")
    val (gbtModel, gbtMetrics, gbtTime) = entrenarGBT(
      trainOptimized, testOptimized
    )
    imprimirMetricas("Gradient Boosted Trees", gbtMetrics, gbtTime)
    imprimirFeatureImportances(gbtModel, "GBT")
    guardarPredicciones(
      gbtModel, 
      testOptimized, 
      "ml_prediction/resultados/simplified_gbt_predictions.txt"
    )
    
    // ============================================================================
    // MODELO 4: Ensemble
    // ============================================================================
    println("\n🎭 PASO 8: Creando modelo ENSEMBLE...")
    val ensembleMetrics = crearEnsemble(
      baselineModel, rfModel, gbtModel, testOptimized
    )
    imprimirMetricas("Ensemble (0.2*Ridge + 0.3*RF + 0.5*GBT)", ensembleMetrics, 0.0)
    
    // ============================================================================
    // REPORTE FINAL
    // ============================================================================
    println("\n📊 PASO 9: Generando reporte comparativo...")
    generarReporteComparativo(
      Map(
        "Ridge Regression" -> (baselineMetrics, baselineTime),
        "Random Forest" -> (rfMetrics, rfTime),
        "Gradient Boosted Trees" -> (gbtMetrics, gbtTime),
        "Ensemble" -> (ensembleMetrics, 0.0)
      )
    )
    
    trainOptimized.unpersist()
    testOptimized.unpersist()
    
    println("\n" + "=" * 80)
    println("✅ PROCESO COMPLETADO - Resultados en ml_prediction/resultados/")
    println("=" * 80)
    
    spark.stop()
  }
  
  // ==========================================================================
  // FUNCIONES DE CARGA Y LIMPIEZA
  // ==========================================================================
  
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
    println(s"   ✓ Dataset completo: ${joinedDF.count()} filas")
    
    joinedDF
  }
  
  def limpiarYEnriquecerDatos(df: DataFrame): DataFrame = {
    val originalCount = df.count()
    
    val dfYearCleaned = df.withColumn(
      "year_clean",
      regexp_replace(col("year"), "[^0-9]", "").cast(IntegerType)
    )
    
    val cleanDF = dfYearCleaned.na.drop(Seq(
      "description", "genre", "director", "actors",
      "duration", "avg_vote", "votes", 
      "reviews_from_users", "reviews_from_critics", "year_clean"
    ))
    
    val cleanCount = cleanDF.count()
    val lossPercent = ((originalCount - cleanCount).toDouble / originalCount * 100)
    
    println(s"   ✓ Filas originales: $originalCount")
    println(s"   ✓ Filas limpias: $cleanCount")
    println(f"   ✓ Pérdida: $lossPercent%.2f%%")
    
    cleanDF
  }
  
  // NUEVA FUNCIÓN: Feature Engineering con split correcto (sin data leakage)
  def aplicarFeatureEngineeringConSplit(
    trainRaw: DataFrame, 
    testRaw: DataFrame
  ): (DataFrame, DataFrame, Map[String, Map[String, Double]]) = {
    
    // 1. Target Encoding para DIRECTOR (solo en train)
    println("   🔧 Aplicando Target Encoding para director (SOLO TRAIN)...")
    val (trainWithDirector, directorMap) = targetEncodeOnTrain(trainRaw, "director", "avg_vote")
    val testWithDirector = applyTargetEncoding(testRaw, "director", directorMap)
    
    // 2. Target Encoding para ACTORS (solo en train)
    println("   🔧 Aplicando Target Encoding para actors (SOLO TRAIN)...")
    val (trainWithActors, actorsMap) = targetEncodeOnTrain(trainWithDirector, "actors", "avg_vote")
    val testWithActors = applyTargetEncoding(testWithDirector, "actors", actorsMap)
    
    // 3. Reducir cardinalidad de GENRE (basado solo en train)
    println("   🔧 Reduciendo cardinalidad de genre (basado en TRAIN)...")
    val (trainWithGenre, genreTopCategories) = reduceCardinalityOnTrain(trainWithActors, "genre", topN = 30)
    val testWithGenre = applyCardinalityReduction(testWithActors, "genre", genreTopCategories)
    
    // 4. Features derivadas (sin leakage - solo usan la misma fila)
    println("   🔧 Creando features derivadas...")
    val trainEnriched = crearFeaturesDerivadasSeguras(trainWithGenre)
    val testEnriched = crearFeaturesDerivadasSeguras(testWithGenre)
    
    println("   ✓ Feature Engineering completado SIN data leakage")
    
    val encodingMaps = Map(
      "director" -> directorMap,
      "actors" -> actorsMap
    )
    
    (trainEnriched, testEnriched, encodingMaps)
  }
  
  // Target Encoding SOLO en train, retorna el mapa de encodings
  def targetEncodeOnTrain(
    df: DataFrame, 
    column: String, 
    targetCol: String
  ): (DataFrame, Map[String, Double]) = {
    
    val globalMean = df.select(mean(targetCol)).first().getDouble(0)
    
    val avgByCategory = df.groupBy(column)
      .agg(
        mean(targetCol).alias("category_mean"),
        count("*").alias("category_count")
      )
      .collect()
    
    val smoothingFactor = 10.0
    val encodingMap = avgByCategory.map { row =>
      val category = row.getString(0)
      val categoryMean = row.getDouble(1)
      val categoryCount = row.getLong(2)
      val smoothedValue = (categoryMean * categoryCount + globalMean * smoothingFactor) / 
                          (categoryCount + smoothingFactor)
      category -> smoothedValue
    }.toMap
    
    // Agregar globalMean como default
    val encodingMapWithDefault = encodingMap + ("__UNKNOWN__" -> globalMean)
    
    val dfEncoded = applyTargetEncoding(df, column, encodingMapWithDefault)
    
    (dfEncoded, encodingMapWithDefault)
  }
  
  // Aplicar Target Encoding usando un mapa pre-calculado
  def applyTargetEncoding(
    df: DataFrame, 
    column: String, 
    encodingMap: Map[String, Double]
  ): DataFrame = {
    
    val defaultValue = encodingMap.getOrElse("__UNKNOWN__", 0.0)
    
    val encodeUDF = udf((value: String) => 
      encodingMap.getOrElse(value, defaultValue)
    )
    
    df.withColumn(s"${column}_encoded", encodeUDF(col(column)))
      .drop(column)
  }
  
  // Reducir cardinalidad SOLO en train
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
  
  // Aplicar reducción de cardinalidad usando categorías pre-calculadas
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
  
  // Features derivadas que NO causan data leakage (solo usan valores de la misma fila)
  def crearFeaturesDerivadasSeguras(df: DataFrame): DataFrame = {
    df
      .withColumn("votes_per_review", 
        col("votes") / (col("reviews_from_users") + col("reviews_from_critics") + 1))
      .withColumn("review_ratio", 
        col("reviews_from_users") / (col("reviews_from_critics") + 1))
      .withColumn("decade", 
        (col("year_clean") / 10).cast(IntegerType) * 10)
      .withColumn("is_recent", 
        when(col("year_clean") >= 2015, 1.0).otherwise(0.0))
      .withColumn("is_old_classic", 
        when(col("year_clean") <= 1980, 1.0).otherwise(0.0))
      .withColumn("log_votes", 
        log1p(col("votes")))
      .withColumn("duration_category",
        when(col("duration") <= 90, "short")
        .when(col("duration") <= 120, "medium")
        .otherwise("long"))
  }
  
  // ==========================================================================
  // CREACIÓN DE PIPELINES
  // ==========================================================================
  
  def crearPipeline(regressor: org.apache.spark.ml.Estimator[_]): Pipeline = {
    // 1. TF-IDF para description (SOLO unigrams, 100 features - REDUCIDO)
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
      .setNumFeatures(100)  // Reducido de 150 a 100
    
    val descIDF = new IDF()
      .setInputCol("desc_tf")
      .setOutputCol("description_features")
    
    // 2. FeatureHasher para genre (16 features - MUY REDUCIDO)
    val genreHasher = new FeatureHasher()
      .setInputCols(Array("genre"))
      .setOutputCol("genre_features")
      .setNumFeatures(16)  // Reducido de 32 a 16
    
    // 3. StringIndexer para duration_category
    val durationIndexer = new StringIndexer()
      .setInputCol("duration_category")
      .setOutputCol("duration_indexed")
      .setHandleInvalid("keep")
    
    // 4. Assembler - TOTAL: ~130 features (REDUCIDO)
    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "description_features",    // 100 features (reducido)
        "genre_features",           // 16 features (reducido)
        "director_encoded",         // 1 feature
        "actors_encoded",           // 1 feature
        "duration",                 // 1 feature
        "duration_indexed",         // 1 feature
        "votes",                    // 1 feature
        "log_votes",                // 1 feature
        "reviews_from_users",       // 1 feature
        "reviews_from_critics",     // 1 feature
        "year_clean",               // 1 feature
        "decade",                   // 1 feature
        "votes_per_review",         // 1 feature
        "review_ratio",             // 1 feature
        "is_recent",                // 1 feature
        "is_old_classic"            // 1 feature
      ))
      .setOutputCol("features")
      .setHandleInvalid("skip")
    
    // 5. Scaler
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
  
  // ==========================================================================
  // ENTRENAMIENTO DE MODELOS
  // ==========================================================================
  
  def entrenarModeloBaseline(trainData: DataFrame, testData: DataFrame): (PipelineModel, Map[String, Double], Double) = {
    println("   ⏳ Entrenando Ridge Regression...")
    
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
    println("   ⏳ Entrenando Random Forest...")
    println("   📊 Parámetros: 30 árboles, profundidad 8, minInstances 10")
    println("   ⚠️  Reducido para evitar OutOfMemory")
    
    val startTime = System.nanoTime()
    
    val rf = new RandomForestRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setNumTrees(30)           // Reducido de 100 a 30
      .setMaxDepth(8)             // Reducido de 10 a 8
      .setMinInstancesPerNode(10) // Aumentado de 5 a 10
      .setSubsamplingRate(0.8)    // Usar solo 80% de datos por árbol
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
    println("   ⏳ Entrenando Gradient Boosted Trees...")
    println("   📊 Parámetros: 50 iteraciones, profundidad 5, stepSize 0.1")
    println("   ⚠️  Reducido para evitar OutOfMemory")
    
    val startTime = System.nanoTime()
    
    val gbt = new GBTRegressor()
      .setLabelCol("avg_vote")
      .setFeaturesCol("scaled_features")
      .setMaxIter(50)      // Reducido de 100 a 50
      .setMaxDepth(5)      // Reducido de 6 a 5
      .setStepSize(0.1)    // Learning rate estándar
      .setSubsamplingRate(0.8) // Usar solo 80% de datos
      .setSeed(42)
    
    val pipeline = crearPipeline(gbt)
    val model = pipeline.fit(trainData)
    
    val endTime = System.nanoTime()
    val elapsedTime = (endTime - startTime) / 1e9
    
    val predictions = model.transform(testData)
    val metrics = evaluarModelo(predictions)
    
    (model, metrics, elapsedTime)
  }
  
  def crearEnsemble(
    baselineModel: PipelineModel,
    rfModel: PipelineModel,
    gbtModel: PipelineModel,
    testData: DataFrame
  ): Map[String, Double] = {
    
    println("   ⏳ Generando predicciones de modelos individuales...")
    
    val baselinePred = baselineModel.transform(testData)
      .select(col("avg_vote"), col("prediction").alias("pred_baseline"))
    
    val rfPred = rfModel.transform(testData)
      .select(col("prediction").alias("pred_rf"))
    
    val gbtPred = gbtModel.transform(testData)
      .select(col("prediction").alias("pred_gbt"))
    
    val combined = baselinePred
      .withColumn("id", monotonically_increasing_id())
      .join(rfPred.withColumn("id", monotonically_increasing_id()), "id")
      .join(gbtPred.withColumn("id", monotonically_increasing_id()), "id")
    
    val ensemble = combined.withColumn(
      "prediction",
      col("pred_baseline") * 0.2 + col("pred_rf") * 0.3 + col("pred_gbt") * 0.5
    )
    
    println("   ✓ Ensemble creado (0.2*Baseline + 0.3*RF + 0.5*GBT)")
    
    guardarPredicciones(
      ensemble.select("avg_vote", "prediction"),
      "ml_prediction/resultados/simplified_ensemble_predictions.txt"
    )
    
    evaluarModelo(ensemble)
  }
  
  // ==========================================================================
  // EVALUACIÓN Y REPORTES
  // ==========================================================================
  
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
    println(s"\n   ✅ Métricas de $modelName:")
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
      case _ => None
    }
    
    treeModel.foreach { importances =>
      println(s"\n   📊 Feature Importances (Top 15):")
      val topFeatures = importances.toArray.zipWithIndex
        .sortBy(-_._1)
        .take(15)
      
      topFeatures.foreach { case (importance, idx) =>
        println(f"      Feature $idx%3d: ${importance * 100}%.2f%%")
      }
    }
  }
  
  def guardarPredicciones(model: PipelineModel, testData: DataFrame, outputPath: String): Unit = {
    val predictions = model.transform(testData)
      .select("avg_vote", "prediction")
      .withColumn("error", abs(col("avg_vote") - col("prediction")))
      .orderBy(desc("error"))
      .limit(20)
    
    guardarPredicciones(predictions, outputPath)
  }
  
  def guardarPredicciones(predictions: DataFrame, outputPath: String): Unit = {
    val predList = predictions.collect()
    
    val writer = new PrintWriter(outputPath)
    writer.println("=" * 80)
    writer.println("TOP 20 PREDICCIONES CON MAYOR ERROR")
    writer.println("=" * 80)
    writer.println(f"${"Real"}%-10s ${"Predicción"}%-12s ${"Error"}%-10s")
    writer.println("-" * 80)
    
    predList.foreach { row =>
      val real = row.getDouble(0)
      val pred = row.getDouble(1)
      val error = math.abs(real - pred)
      writer.println(f"$real%-10.4f $pred%-12.4f $error%-10.4f")
    }
    
    writer.println("=" * 80)
    writer.close()
    
    println(s"   ✓ Predicciones guardadas en: $outputPath")
  }
  
  def generarReporteComparativo(modelos: Map[String, (Map[String, Double], Double)]): Unit = {
    val outputPath = "ml_prediction/resultados/reporte_simplificado.txt"
    val writer = new PrintWriter(outputPath)
    
    writer.println("=" * 80)
    writer.println("REPORTE COMPARATIVO - MODELOS DE PREDICCIÓN IMDB")
    writer.println("=" * 80)
    writer.println()
    
    writer.println("CONFIGURACIÓN:")
    writer.println("-" * 80)
    writer.println("• Features de Texto:")
    writer.println("  - Description: TF-IDF Unigrams (100 features)")
    writer.println("• Features Categóricas:")
    writer.println("  - Genre: Feature Hashing (16 features) - Top 30 categorías")
    writer.println("  - Director: Target Encoding (1 feature)")
    writer.println("  - Actors: Target Encoding (1 feature)")
    writer.println("  - Duration Category: StringIndexer (1 feature)")
    writer.println("• Features Numéricas (11 features):")
    writer.println("  - duration, votes, log_votes, reviews_from_users, reviews_from_critics")
    writer.println("  - year_clean, decade, votes_per_review, review_ratio")
    writer.println("  - is_recent, is_old_classic")
    writer.println("• TOTAL: ~130 features (Optimizado para memoria)")
    writer.println("• Normalización: StandardScaler")
    writer.println("• Random Forest: 30 árboles, profundidad 8, subsample 0.8")
    writer.println("• GBT: 50 iteraciones, profundidad 5, subsample 0.8")
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
    val baselineMetrics = modelos("Ridge Regression")._1
    
    writer.println("ANÁLISIS:")
    writer.println("-" * 80)
    writer.println(s"🏆 Mejor modelo: ${mejorModelo._1}")
    writer.println(f"   - RMSE: ${mejorModelo._2._1("RMSE")}%.4f")
    writer.println(f"   - R²: ${mejorModelo._2._1("R2")}%.4f")
    writer.println()
    
    val mejoriaRMSE = ((baselineMetrics("RMSE") - mejorModelo._2._1("RMSE")) / baselineMetrics("RMSE") * 100)
    val mejoriaR2 = ((mejorModelo._2._1("R2") - baselineMetrics("R2")) / baselineMetrics("R2") * 100)
    
    writer.println(s"📈 Mejora sobre baseline:")
    writer.println(f"   - RMSE: $mejoriaRMSE%.2f%% mejor")
    writer.println(f"   - R²: $mejoriaR2%.2f%% mejor")
    writer.println()
    
    writer.println("TÉCNICAS APLICADAS:")
    writer.println("-" * 80)
    writer.println("✅ Target Encoding (director, actors) - Evita alta cardinalidad")
    writer.println("✅ Feature Hashing (genre) - Memoria controlada")
    writer.println("✅ Feature Engineering - 11 features derivadas")
    writer.println("✅ TF-IDF - Captura semántica del texto")
    writer.println("✅ StandardScaler - Normalización de features")
    writer.println("✅ Ensemble - Combina fortalezas de múltiples modelos")
    writer.println()
    
    writer.println("=" * 80)
    writer.close()
    
    println(s"   ✓ Reporte guardado en: $outputPath")
    
    println("\n" + "=" * 80)
    println("📊 RESUMEN FINAL")
    println("=" * 80)
    println(f"${"Modelo"}%-35s ${"RMSE"}%-10s ${"R²"}%-10s")
    println("-" * 80)
    modelos.toSeq.sortBy(_._2._1("RMSE")).foreach { case (nombre, (metricas, _)) =>
      println(f"$nombre%-35s ${metricas("RMSE")}%-10.4f ${metricas("R2")}%-10.4f")
    }
    println("=" * 80)
    println(s"🏆 Mejor modelo: ${mejorModelo._1}")
    println(f"📈 Mejora sobre baseline: RMSE ${mejoriaRMSE}%.2f%% | R² ${mejoriaR2}%.2f%%")
    println("=" * 80)
  }
}
