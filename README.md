## Redes adversarias generativas (GANs)

### 1. **Objetivo General**

Las GANs buscan aprender la distribución de probabilidad de los datos reales $p_{\text{data}}(x)$, usando una red generadora $G(z)$, que transforma ruido $z \sim p_z(z)$ (por ejemplo, una gaussiana) en muestras sintéticas $G(z) \sim p_g(x)$.

---

### 2. **Componentes del Modelo**

#### 🔹 Generador $G(z; \theta_g)$

* Toma como entrada una variable aleatoria $z \sim p_z(z)$, típicamente ruido gaussiano o uniforme.
* Devuelve una muestra sintética $x' = G(z)$.
* Tiene parámetros $\theta_g$ que se actualizan para que $p_g(x) \approx p_{\text{data}}(x)$.

#### 🔹 Discriminador $D(x; \theta_d)$

* Es una función que estima la probabilidad de que una muestra $x$ provenga del conjunto de datos reales.
* Devuelve un valor en $[0, 1]$:

  $$
  D(x) = \mathbb{P}(x \text{ es real})
  $$
* Se entrena para maximizar su capacidad de distinguir entre datos reales y sintéticos.

---

### 3. **Juego Minimax (Formulación de Teoría de Juegos)**

La GAN se entrena como un **juego de dos jugadores** en el que el generador quiere engañar al discriminador, y este último quiere detectarlo:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

#### 🔍 Interpretación:

* $\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]$: El discriminador es recompensado por asignar alta probabilidad a los datos reales.
* $\mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$: Es recompensado por asignar baja probabilidad a datos generados.
* El generador quiere minimizar la segunda expectativa, es decir, quiere que $D(G(z)) \to 1$.

---

### 4. **Entrenamiento Alternado**

1. **Actualizar $D$**:
   Mantener $G$ fijo, y maximizar:

   $$
   \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
   $$

2. **Actualizar $G$**:
   Mantener $D$ fijo, y minimizar:

   $$
   \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
   $$

   O mejor (para evitar gradientes débiles cuando $D(G(z)) \approx 0$):

   $$
   \text{Usar: } \mathbb{E}_{z \sim p_z}[-\log D(G(z))] \quad \text{(Heuristic non-saturating loss)}
   $$


### 5. **GAN Lab**

En el siguiente enlace se encuentra disponible una ilustración interactiva del funcionamiento de las (GANs): https://poloclub.github.io/ganlab/

Aquí tienes una **sección para agregar al final del `README.md`** con la descripción clara de la tarea para estudiantes del curso de Redes Neuronales y Aprendizaje Profundo. Está redactada de forma académica, con énfasis en la comprensión del entrenamiento de la GAN y en el desarrollo de una interfaz gráfica.

---

### 6. **Tarea: Análisis de Dinámica de Entrenamiento y Desarrollo de Interfaz Interactiva**

A partir del script `run_experiment.py`, los estudiantes deben desarrollar una actividad práctica que consta de dos partes:

#### Parte 1: Análisis de la dinámica de entrenamiento

1. Ejecutar el script `run_experiment.py` para entrenar una red generativa adversaria sobre el conjunto de datos MNIST.
2. Analizar la evolución de las pérdidas del generador y del discriminador a lo largo de las épocas.
3. Realizar un **reporte técnico** que incluya:

   * Gráficos de las pérdidas y su interpretación.
   * Discusión sobre el equilibrio entre generador y discriminador.
   * Visualización de las imágenes generadas en distintas épocas.
   * Reflexión sobre la convergencia y estabilidad del modelo.

#### Parte 2: Interfaz gráfica con generación y reconocimiento

Desarrollar una **interfaz gráfica** que incluya al menos dos funcionalidades:

* **Generar dígito aleatorio**: Un botón que, al hacer clic, utilice el generador entrenado para crear una nueva imagen de dígito.
* **Reconocer dígito generado**: Otro botón que aplique un clasificador entrenado (por ejemplo, un modelo preentrenado en PyTorch) para identificar automáticamente qué dígito representa la imagen generada. Opcionalmente, se pueden mostrar los valores de confianza (probabilidades) del clasificador.