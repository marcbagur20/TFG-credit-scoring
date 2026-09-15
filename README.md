# Credit scoring con GLM, SMOTE y Random Forest

Código y resultados del Trabajo de Fin de Grado *Modelos estadísticos clásicos y avanzados. Aplicación al credit scoring utilizando Modelos Lineales Generalizados, SMOTE y Árboles de Decisión*.

Grado en Economía · Universidad de Alcalá · Curso 2025/2026
Autor: Marc Bagur Vizoso · Tutor: David Atance del Olmo

---

## Qué hay aquí

Se comparan **siete modelos de predicción de impago** sobre 112.498 préstamos al consumo de la plataforma Prosper, bajo la misma partición, la misma semilla y el mismo umbral de decisión:

| Familia | Modelos |
|---|---|
| GLM | Logit *backward*, Logit *forward*, Probit *backward*, Probit *forward* |
| Remuestreo | Logit + SMOTE |
| Ensamble | Random Forest, Random Forest + SMOTE |

La pregunta no es cuál predice mejor, sino qué se gana y qué se pierde en cada caso: capacidad de ordenar el riesgo, capacidad de detectar impagos e interpretabilidad del modelo.

## Resultados sobre la muestra de prueba

22.500 observaciones no utilizadas en la estimación, umbral de clasificación de 0,5.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logit *backward* | 0,8632 | 0,6502 | 0,2766 | 0,3881 | 0,8591 |
| Logit *forward* | 0,8633 | 0,6507 | 0,2771 | 0,3887 | 0,8591 |
| Probit *backward* | 0,8635 | 0,6610 | 0,2658 | 0,3791 | 0,8594 |
| Probit *forward* | 0,8632 | 0,6594 | 0,2644 | 0,3774 | 0,8594 |
| Logit + SMOTE | 0,7425 | 0,3593 | **0,8189** | **0,4994** | 0,8576 |
| Random Forest | 0,8605 | **0,6653** | 0,2230 | 0,3340 | **0,8643** |
| Random Forest + SMOTE | 0,8604 | 0,5929 | 0,3508 | 0,4408 | 0,8590 |

Tres lecturas:

1. **La capacidad discriminante es prácticamente la misma en los siete modelos.** Entre el mejor y el peor ROC-AUC hay cinco milésimas. Random Forest gana, pero a costa de la interpretabilidad, que en un sector supervisado no es un coste menor.
2. **La capacidad de detección no lo es en absoluto.** El recall va de 0,2230 a 0,8189 sobre los mismos datos y con el mismo umbral. De los 3.529 impagos de la muestra de prueba, los GLM detectan menos de mil; el modelo con SMOTE identifica 2.890, a cambio de marcar como dudosas 5.154 operaciones solventes.
3. **El ajuste dentro de muestra no sirve para comparar modelos.** Las dos variantes de Random Forest alcanzan métricas perfectas en entrenamiento (árboles sin restricción de profundidad) y son las peores detectando impagos fuera de muestra.

## Estructura del repositorio

```
notebooks/   Análisis completo en Python (Google Colab)
src/         Scripts auxiliares
figures/     Figuras generadas para la memoria
results/     Tablas y métricas exportadas
data/        Instrucciones para obtener el dataset (no se incluye por tamaño)
```

## Cómo reproducirlo

1. Descarga el dataset original de Kaggle: [Prosper Loan Data](https://www.kaggle.com/datasets/henryokam/prosper-loan-data). No se incluye aquí por tamaño; consulta `data/README.md`.
2. Abre el notebook en Colab:

   [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcbagur20/TFG-credit-scoring/blob/main/notebooks/TFG_credit_scoring.ipynb)

3. Sube `prosperLoanData.csv` al entorno o ajusta la variable `DATA_PATH` de la primera celda.
4. Ejecuta las celdas en orden.

En local: `pip install -r requirements.txt`.

Hay dos versiones del mismo notebook. `TFG_credit_scoring.ipynb` conserva todos los resultados y figuras, pero pesa 4 MB y GitHub no lo previsualiza; ábrelo en Colab. `TFG_credit_scoring_sin_resultados.ipynb` es el mismo código sin salidas y sí se lee directamente en el navegador.

## Decisiones metodológicas relevantes

- **Variable objetivo.** Se consideran impago los préstamos en estado *Chargedoff* o *Defaulted* y los que presentan mora superior a 60 días. Tasa resultante: 15,68 %.
- **Partición.** 80/20 aleatoria y estratificada por la variable dependiente, con semilla fija (`RANDOM_STATE = 42`).
- **SMOTE.** Se aplica dentro de un `Pipeline` de `imbalanced-learn`, de modo que las observaciones sintéticas se generan **solo sobre la partición de entrenamiento**. El conjunto de prueba conserva la distribución original de clases.
- **Selección de variables.** Procedimientos *forward* y *backward stepwise* por significatividad individual (p < 0,05), ejecutados únicamente sobre la muestra de entrenamiento. AIC y BIC se usan para comparar las especificaciones finales, no como regla de parada.
- **Variables excluidas.** Identificadores, variables posteriores a la concesión (`LoanCurrentDaysDelinquent`, `LP_NetPrincipalLoss`) y calificaciones internas de la plataforma, para evitar fuga de información y estimar solo con datos disponibles en el momento de evaluar la solicitud.

## Limitaciones conocidas

- **Censura por la derecha.** El 59,6 % de las operaciones clasificadas como no impago seguían vigentes en el momento de recogida de los datos, por lo que los modelos capturan también un componente de antigüedad de la operación. El tratamiento correcto sería un modelo de supervivencia.
- **Partición única.** No se emplea validación cruzada ni remuestreo, de modo que los coeficientes y las métricas están condicionados a una división concreta de la muestra, y la inferencia no incorpora la variabilidad del propio proceso de selección secuencial.
- **Fuga menor.** La variable `State_high_risk` se codifica a partir de la tasa histórica de impago calculada sobre la muestra completa, antes de la partición.
- **Random Forest sin ajuste de hiperparámetros**, lo que explica el ajuste perfecto dentro de muestra.
- **Validez externa.** Los datos proceden de una plataforma estadounidense de préstamos entre particulares, cuyo perfil de solicitantes y criterios de concesión difieren de los de una entidad bancaria tradicional.

## Datos

Okam, H. (2024). *Prosper Loan Data*. Kaggle. Operaciones originadas entre 2005 y 2014: 113.937 registros y 81 variables en origen; 112.498 observaciones y 42 variables explicativas tras el preprocesamiento.

## Stack

Python · pandas · NumPy · scikit-learn · statsmodels · imbalanced-learn · Matplotlib · seaborn

## Licencia

Código publicado bajo licencia MIT. El dataset pertenece a sus autores originales y está sujeto a los términos de Kaggle.
