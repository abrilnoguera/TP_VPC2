# Detección Automatizada de Cáncer de Pulmón mediante Visión por Computadora

## Información del Proyecto

**Materia:** Visión por Computadora II - CEIA  
**Autores:** Abril Noguera - Pedro Barrera - Ezequiel Caamaño  
**Fecha:** Octubre 2025  

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Business Case](#business-case)
3. [Dataset](#dataset)
4. [Metodología](#metodología)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Experimentos Realizados](#experimentos-realizados)
7. [Resultados](#resultados)
8. [Conclusiones](#conclusiones)
9. [Instalación y Uso](#instalación-y-uso)
10. [Trabajo Futuro](#trabajo-futuro)

## Descripción del Proyecto

Este proyecto desarrolla un sistema inteligente basado en redes neuronales convolucionales (CNN) para la detección automatizada de cáncer de pulmón mediante análisis de imágenes de tomografías computarizadas (CT-Scans) de tórax.

### Objetivos

El sistema es capaz de:
- **Detectar** si el paciente presenta signos de cáncer pulmonar
- **Clasificar** el tipo específico de cáncer entre:
  - **Adenocarcinoma**: ~30% de todos los casos, ~40% de NSCLC
  - **Carcinoma de células grandes**: 10-15% de NSCLC
  - **Carcinoma de células escamosas**: localizado en bronquios principales
  - **Normal**: casos sin evidencia de cáncer

## Business Case

### Motivación

El cáncer de pulmón es una de las principales causas de muerte a nivel mundial. La detección temprana es crucial para mejorar el pronóstico, pero los métodos tradicionales dependen de interpretación humana, lo que puede generar:
- Errores de diagnóstico
- Demoras en el proceso
- Variabilidad entre profesionales

### Impacto Esperado

**Impacto Clínico:**
- Reducción del tiempo diagnóstico
- Mayor precisión y objetividad
- Herramienta complementaria para especialistas

**Impacto Social:**
- Mejora en calidad de vida por detección temprana
- Democratización del acceso al diagnóstico

**Impacto Económico:**
- Reducción de costos por diagnósticos tardíos
- Optimización de recursos hospitalarios

## Dataset

**Fuente:** Chest CT-Scan Images (Kaggle)

**Estructura:**
- **Clases:** 4 categorías (Normal, Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma)
- **División:** Train/Validation/Test
- **Formato:** Imágenes CT-Scan en formato PNG/JPEG

**Análisis Exploratorio:**
- Distribución balanceada entre clases
- Análisis de características de imágenes
- Evaluación de calidad y variabilidad

## Metodología

### Enfoque General
- **Transfer Learning** como estrategia principal
- **ResNet18** preentrenada en ImageNet como arquitectura base
- Evaluación progresiva de estrategias de mejora

### Métricas de Evaluación
- **Accuracy**: Precisión general del modelo
- **Precision, Recall, F1-Score**: Por clase y macro-promedio
- **Matriz de Confusión**: Análisis detallado de errores
- **Sensibilidad**: Métrica crítica en diagnóstico médico

## Estructura del Proyecto

```
TP_VPC2/
├── BC & EDA.ipynb              # Business Case y Análisis Exploratorio
├── Baseline.ipynb             # Modelo base de referencia
├── Exploration.ipynb          # Estrategias avanzadas de mejora
├── Output/                    # Modelos entrenados
│   ├── baseline_model.pth
│   ├── modelo_conservador.pth
│   ├── modelo_unfrozen.pth
│   ├── modelo_2_etapas.pth
│   └── *_best.pth            # Mejores versiones de cada modelo
└── README.md                  # Este archivo
```

## Experimentos Realizados

### 1. Modelo Baseline

**Configuración:**
- Arquitectura: ResNet18 preentrenada
- Estrategia: Transfer Learning básico
- Capas congeladas: Todas las convolucionales
- Entrenamiento: Solo capa final (FC)

**Parámetros:**
- Batch Size: 32
- Epochs: 5
- Learning Rate: 1e-3
- Optimizer: Adam

**Resultados:**
- Buen rendimiento en diferenciación normal vs patológico
- Limitaciones en separación entre subtipos de cáncer

### 2. Estrategia 1: Data Augmentation Conservador

**Justificación:** Mejorar robustez manteniendo simplicidad del baseline

**Técnicas aplicadas:**
- Rotaciones leves (≤5°)
- Cambios de contraste moderados
- Flip horizontal
- Transformaciones específicas para CT médicos

**Mejoras:**
- Aumento en sensibilidad para Large Cell Carcinoma (+0.17)
- Mayor robustez sin afectar estabilidad
- Reducción de confusiones entre subtipos

### 3. Estrategia 2: Fine-Tuning + Data Augmentation Moderado

**Objetivo:** Aumentar capacidad de especialización del modelo

**Técnicas aplicadas:**
- Descongelamiento de layer4 de ResNet18
- Data augmentation más agresivo
- Uso de Albumentations para transformaciones médicas

**Configuración:**
- Epochs: 20
- Learning rates diferenciados (FC: 1e-3, layer4: 1e-4)
- Scheduler: ReduceLROnPlateau

**Resultados:**
- Mejoras significativas en todas las clases
- Precisión >0.8 en todas las categorías
- Reducción drástica de confusiones entre subtipos

### 4. Estrategia 3: Entrenamiento en 2 Etapas

**Objetivo:** Maximizar estabilidad y convergencia

**Metodología:**
- **Etapa 1:** Transfer learning básico (10 epochs)
- **Etapa 2:** Fine-tuning progresivo con layer4 descongelado (15 epochs)

**Ventajas:**
- Proceso más controlado y estable
- Convergencia óptima de capa FC antes del fine-tuning
- Minimización del riesgo de corrupción de pesos preentrenados

## Resultados

### Comparación de Modelos

| Modelo | Accuracy | F1-Score Macro | Observaciones |
|--------|----------|----------------|---------------|
| Baseline | 67.03% | 67.03% | Rápido, limitado en subtipos |
| Conservador | 75.25% | 75.25% | Mejora marginal con augmentation |
| Unfrozen | 86.14% | 86.14% | Mayor especialización |
| 2 Etapas | **87.69%** | **87.69%** | Mejor estabilidad y performance |

### Hallazgos Clave

1. **Transfer Learning es efectivo** para imágenes médicas CT
2. **Fine-tuning es crucial**: El salto más significativo se da al descongelar capas convolucionales
3. **Entrenamiento progresivo es superior**: La Estrategia 3 ofrece la mejor combinación de performance y estabilidad
4. **Data augmentation específico** para CT-scans mejora robustez

### Modelo Final

**Estrategia 3 (2 Etapas)** seleccionada como modelo de producción:
- **F1-Score:** 87.69%
- **Accuracy:** 87.69%
- **Metodología más estable**
- **Mejor balance entre todas las clases**

## Conclusiones

### Logros Principales

1. **Sistema funcional** para detección de cáncer de pulmón con alta precisión
2. **Comparación sistemática** de diferentes estrategias de transfer learning
3. **Mejora progresiva** desde baseline hasta modelos especializados
4. **Metodología reproducible** y bien documentada

### Limitaciones Identificadas

- Necesidad de validación clínica con especialistas
- Tamaño limitado del dataset
- Variabilidad en calidad de imágenes CT
- Falta de análisis de explicitabilidad del modelo

### Contribuciones Técnicas

- Implementación de estrategias progresivas de fine-tuning
- Uso de Albumentations para data augmentation específico en imágenes médicas
- Comparación sistemática de métricas críticas para aplicaciones médicas
- Desarrollo de pipeline completo desde EDA hasta evaluación

## Instalación y Uso

### Requisitos

```python
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.3.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
numpy>=1.21.0
PIL>=8.3.0
kagglehub
```

### Uso Básico

1. **Clonar el repositorio:**
```bash
git clone https://github.com/abrilnoguera/TP_VPC2.git
cd TP_VPC2
```

2. **Configurar dataset:**
```python
# Actualizar variable de entorno con ruta del dataset
export DATASET_PATH="/path/to/your/dataset"
```

3. **Ejecutar notebooks:**
```bash
# Business Case y EDA
jupyter notebook "BC & EDA.ipynb"

# Modelo Baseline
jupyter notebook "Baseline.ipynb"

# Estrategias avanzadas
jupyter notebook "Exploration.ipynb"
```

4. **Cargar modelo entrenado:**
```python
import torch
from torchvision import models

# Cargar mejor modelo
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("Output/Etapa_2_best.pth"))
model.eval()
```

## Trabajo Futuro

### Mejoras Propuestas

**Datos:**
- Aumento del dataset con más casos clínicos
- Balanceo específico de clases
- Validación cruzada con otros hospitales/datasets

**Modelos:**
- Evaluación de arquitecturas más complejas (ResNet50, EfficientNet, Vision Transformers)
- Implementación de ensemble methods
- Técnicas de segmentación previa de regiones de interés

**Validación Clínica:**
- Colaboración con radiólogos para validación
- Estudios de usabilidad en entorno hospitalario
- Análisis de explicitabilidad (Grad-CAM, LIME)

**Implementación:**
- Desarrollo de API REST para integración hospitalaria
- Interface web para radiólogos
- Optimización para inferencia en tiempo real

### Extensiones Potenciales

- Detección de otros tipos de cáncer
- Análisis temporal de progresión
- Integración con sistemas PACS hospitalarios
- Desarrollo de herramientas de explicitabilidad

## Licencia

Este proyecto está desarrollado con fines académicos para la materia Visión por Computadora II de la Carrera de Especialización en Inteligencia Artificial (CEIA).

## Contacto

**Equipo de Desarrollo:**
- Abril Noguera
- Pedro Barrera  
- Ezequiel Caamaño

**Programa:** CEIA - Visión por Computadora II

---

*Este proyecto demuestra el potencial de la inteligencia artificial en medicina, específicamente en la asistencia al diagnóstico médico y la democratización de herramientas especializadas.*