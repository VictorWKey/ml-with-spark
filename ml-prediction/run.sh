#!/bin/bash

# ============================================================================
# Script de ejecución - Modelo de Predicción IMDB
# ============================================================================

echo "🎬 IMDB Rating Prediction - Modelo de Machine Learning"
echo "========================================================"
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar si Spark está instalado
if ! command -v spark-shell &> /dev/null; then
    echo "❌ Error: spark-shell no encontrado"
    echo "   Instala Apache Spark primero"
    exit 1
fi

echo -e "${GREEN}✓${NC} Spark encontrado"
echo ""

# Mostrar opciones
echo "Selecciona una opción:"
echo "  1) Ejecutar modelo completo (Ridge + RF + GBT + Ensemble)"
echo "  2) Solo validar datos"
echo "  3) Ver resultados existentes"
echo "  4) Limpiar resultados y re-ejecutar"
echo ""
read -p "Opción [1-4]: " option

case $option in
    1)
        echo ""
        echo -e "${YELLOW}Iniciando Spark Shell con 14GB de memoria...${NC}"
        echo "Esto puede tomar 1-2 minutos para iniciar"
        echo ""
        
        spark-shell \
          --driver-memory 14g \
          --executor-memory 14g \
          --conf spark.memory.fraction=0.8 \
          --conf spark.memory.storageFraction=0.2 \
          --conf spark.sql.shuffle.partitions=50 \
          --conf spark.driver.maxResultSize=2g \
          -i <(echo ':load ml_prediction/IMDBPredictionModelSimplified.scala'; \
               echo 'IMDBPredictionModelSimplified.main(Array())'; \
               echo ':quit')
        ;;
    
    2)
        echo ""
        echo -e "${YELLOW}Ejecutando validación de datos...${NC}"
        echo ""
        
        spark-shell \
          --driver-memory 8g \
          --executor-memory 8g \
          -i <(echo ':load ml_prediction/DataValidation.scala'; \
               echo 'DataValidation.main(Array())'; \
               echo ':quit')
        ;;
    
    3)
        echo ""
        echo -e "${GREEN}Resultados existentes:${NC}"
        echo ""
        ls -lh ml_prediction/resultados/
        echo ""
        echo "Reporte principal:"
        cat ml_prediction/resultados/reporte_simplificado.txt
        ;;
    
    4)
        echo ""
        echo -e "${YELLOW}⚠️  ¿Estás seguro de eliminar resultados existentes?${NC}"
        read -p "Escribe 'SI' para confirmar: " confirm
        
        if [ "$confirm" = "SI" ]; then
            rm -f ml_prediction/resultados/simplified_*.txt
            rm -f ml_prediction/resultados/reporte_simplificado.txt
            echo -e "${GREEN}✓${NC} Resultados eliminados"
            echo ""
            echo "Re-ejecutando modelo..."
            $0 1  # Recursively call option 1
        else
            echo "Cancelado"
        fi
        ;;
    
    *)
        echo "Opción inválida"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Proceso completado${NC}"
echo ""
