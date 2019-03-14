# E12 - Gradient Boosting Review

Search for and comment about the main differences between the algorithms implemented in: 

(1) [ Gradient Boosting Classifier ](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

(2) [ XGB Classifier ](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

Con el fin de comprender e identificar diferencias entre el Gradient Boosting Classifier y el Extreme Gradient Boosting (XBG) Classifier, es importante aclarar que ninguno de estos es propiamente un algoritmo de Machine Learning. En sí, estos dos clasificadores se pueden entender bajo el concepto de "Meta-algoritmo", el cual se aplica a un conjunto de modelos de Machine Learning. Usualmente, este tipo de meta-algoritmos usan técnicas de ensamblajes (ensemble) para incrementar la capacidad de aprendizaje de los modelos individualmente y de esta forma crear un único modelo "fuerte" basado en eso modelos un poco más "débiles". 

Con base en estos conceptos, pasamos a explicar las diferencias que presentan estos clasificadores en la práctica. Por un lado, Gradient Boosting Classifier surge como la respuesta a encontrar un modelo predictivo que reuniera la capacidad predictiva de un conjunto de modelos basado en el algoritmo de gradiente descendente. De esta forma, por medio de modelos árboles de decisión se creó el ensamblaje que da origen al Gradient Boosting. Por otro lado, el Extreme Gradient Boosting fue creado igual que el Gradient Boosting, con la diferencia de mantener un enfoque en el desempeño computacional de los modelos. Dado que el Gradient Boosting se basa en árboles de decisión que pueden emplear gran parte de los recursos disponibles en las máquinas, se vio la necesidad de crear un modelo escalabale, manejable en una librería y preciso. Haciendo énfasis en la precisión, se evidenció que el Gradient Boosting tendía a hacer sobre ajustamiento de los datos, con lo cual las predicciones tenian un sesgo y reducían la precisión. Por lo tanto, el crear del nuevo modelo, Tianqi Chen, propuso un nuevo modelo más regularizado que lograra controlar el sobre ajuste de los datos y mejorar el desempeño.

Adicional a esto, XGB se plantea como un modelo en el cual no se aconseja trabajar con imagenes, visión computacional, procesamiento y entendimiento del lenguaje natural o cuando tengamos una muestra de entrenamiento muy pequeña, dada su naturaleza de enfocarse en problemas de regresión.
