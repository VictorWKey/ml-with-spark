import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

// Cargar en Spark Shell:
// :load ml_prediction/DataValidation.scala
// DataValidation.main(Array())

object DataValidation {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("IMDB Data Validation")
      .master("local[*]")
      .config("spark.driver.memory", "6g")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    println("=" * 80)
    println("VALIDACIÓN DE DATOS - MODELO DE PREDICCIÓN IMDB")
    println("=" * 80)
    println()
    
    val moviesPath = "IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb movies.csv"
    val ratingsPath = "IMDB-Movies-Extensive-Dataset-Analysis/data1/IMDb ratings.csv"
    
    // PASO 1: Cargar datos
    println("📊 PASO 1: Cargando datos...")
    val moviesDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("escape", "\"")
      .csv(moviesPath)
    
    val ratingsDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(ratingsPath)
    
    println(s"   ✓ Movies: ${moviesDF.count()} filas")
    println(s"   ✓ Ratings: ${ratingsDF.count()} filas")
    
    // PASO 2: Join
    println("\n📊 PASO 2: Haciendo JOIN...")
    val fullDF = moviesDF.join(ratingsDF, Seq("imdb_title_id"), "left")
    println(s"   ✓ Dataset completo: ${fullDF.count()} filas, ${fullDF.columns.length} columnas")
    
    // PASO 3: Seleccionar columnas que necesitamos
    println("\n📊 PASO 3: Seleccionando columnas para el modelo...")
    val selectedDF = fullDF.select(
      col("imdb_title_id"),
      col("title"),
      col("year"),
      col("genre"),
      col("duration"),
      col("director"),
      col("writer"),
      col("production_company"),
      col("actors"),
      col("description"),
      col("avg_vote").as("label"),
      col("votes"),
      col("reviews_from_users"),
      col("reviews_from_critics")
    )
    
    println(s"   ✓ Columnas seleccionadas: ${selectedDF.columns.length}")
    println(s"   ✓ Total filas: ${selectedDF.count()}")
    
    // PASO 4: Análisis de nulos en columnas clave
    println("\n📊 PASO 4: Analizando valores nulos en columnas clave...")
    val columnas = Array(
      "label", "genre", "director", "actors", "description", 
      "duration", "votes", "reviews_from_users", "reviews_from_critics", 
      "year", "writer", "production_company"
    )
    
    columnas.foreach { col_name =>
      val nullCount = selectedDF.filter(col(col_name).isNull || col(col_name) === "").count()
      val total = selectedDF.count()
      val percent = (nullCount.toDouble / total * 100)
      println(f"   - $col_name%-25s: $nullCount%6d nulos (${percent}%6.2f%%)")
    }
    
    // PASO 5: Limpiar year (convertir a numérico)
    println("\n📊 PASO 5: Limpiando columna 'year'...")
    val withCleanYear = selectedDF.withColumn(
      "year_clean",
      when(col("year").cast("int").isNotNull, col("year").cast("int"))
        .otherwise(lit(null))
    ).drop("year")
    
    val yearNulls = withCleanYear.filter(col("year_clean").isNull).count()
    println(s"   ✓ Valores de 'year' que no son numéricos: $yearNulls")
    
    // PASO 6: Eliminar nulos en columnas CRÍTICAS (las que incluiremos en el modelo)
    println("\n📊 PASO 6: Eliminando filas con nulos en columnas críticas...")
    val cleanDF = withCleanYear
      .na.drop(Seq(
        "label",          // avg_vote
        "genre",
        "director",
        "actors",
        "description",
        "duration",
        "votes",
        "reviews_from_users",
        "reviews_from_critics",
        "year_clean"
      ))
      .filter(
        col("genre") =!= "" &&
        col("director") =!= "" &&
        col("actors") =!= "" &&
        col("description") =!= ""
      )
    
    val originalCount = selectedDF.count()
    val cleanCount = cleanDF.count()
    val removedCount = originalCount - cleanCount
    val percentLost = (removedCount.toDouble / originalCount * 100)
    
    println(s"   ✓ Filas originales: $originalCount")
    println(s"   ✓ Filas después de limpieza: $cleanCount")
    println(s"   ✓ Filas eliminadas: $removedCount (${f"$percentLost%.2f"}%%)")
    
    // PASO 7: Estadísticas de la variable objetivo (avg_vote/label)
    println("\n📊 PASO 7: Estadísticas de la variable objetivo (avg_vote)...")
    cleanDF.select("label").describe().show()
    
    // PASO 8: Distribución de ratings
    println("\n📊 PASO 8: Distribución de ratings...")
    cleanDF.groupBy((col("label") * 10).cast("int") / 10.0)
      .count()
      .orderBy("label")
      .show(20)
    
    // PASO 9: Muestra de datos limpios
    println("\n📊 PASO 9: Muestra de datos limpios (5 registros)...")
    cleanDF.select("title", "year_clean", "genre", "label", "votes")
      .show(5, truncate = false)
    
    // PASO 10: Resumen final
    println("\n" + "=" * 80)
    println("✅ RESUMEN DE VALIDACIÓN")
    println("=" * 80)
    println(f"Total de películas disponibles para entrenamiento: $cleanCount%,d")
    println(f"Porcentaje de datos utilizables: ${(cleanCount.toDouble / originalCount * 100)}%.2f%%")
    println()
    
    if (cleanCount > 10000) {
      println("✅ Tenemos suficientes datos para entrenar el modelo completo")
      println(s"   Recomendación: Entrenar con todos los datos ($cleanCount registros)")
    } else if (cleanCount > 5000) {
      println("⚠️  Datos limitados pero suficientes para entrenamiento")
      println(s"   Recomendación: Entrenar con precaución ($cleanCount registros)")
    } else {
      println("❌ Pocos datos disponibles")
      println(s"   Recomendación: Revisar criterios de limpieza ($cleanCount registros)")
    }
    
    println()
    println("Siguiente paso: Ejecutar IMDBPredictionModel.scala para entrenar modelos")
    println("=" * 80)
    
    spark.stop()
  }
}
