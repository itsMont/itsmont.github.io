---
layout: post
title:  "Research Internship at Purdue University – HPC Semiconductor Industry Implementations"
date:   2026-11-18 17:20:59 -0500
categories: [hpc, mpi, cuda]
---
# Introducción  

Este trabajo de grado tiene el objetivo de realizar una exploración de las aplicaciones de la computación de alto rendimiento en sectores modernos de la industria aprovechando mi estancia de investigación en el laboratorio del profesor Tillmann Kubis gracias al programa [**UREP-C**](https://www.purdue.edu/colombia/index.php) en cooperación con Purdue University. Recorre cuatro problemas: ordenamiento paralelo con MPI, multiplicación e
inversión de matrices en NEMO5, el paso de *singletons* a sesiones RAII para convertir NEMO5 en librería, y el entrenamiento de una red neuronal en CUDA.


<!-- Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %} -->

<!-- MathJax 3: kramdown convierte $$...$$ en \(...\) y \[...\], que MathJax renderiza.
     Si prefieres cargarlo en todo el sitio, mueve estas dos líneas a _includes/head.html. -->
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>


* TOC
{:toc}

## Reseña histórica de Purdue

<!-- Pega aquí el contenido de resena_purdue.tex -->

## Grupo y proyecto de investigación

### Programa UREP-C y The Kubis Group

Trabajé en [**The Kubis Group**](https://www.group.tkubis.com/members), liderado por el Dr. Tillmann Kubis, en la Facultad de Ingeniería
Eléctrica y Computacional (ECE) de Purdue. La línea de trabajo es el desarrollo, la refactorización y la evaluación de rendimiento de módulos HPC en la suite de simulación de semiconductores **NEMO5/RHINO**, y la creación de la librería **QCL**.

### NEMO5/RHINO y la librería QCL

**NEMO5/RHINO** es una herramienta paralela, multiescala y multifísica para modelar dispositivos
nanoelectrónicos y semiconductores. Integra:

- Estructura electrónica (*Tight-Binding*).
- Transporte cuántico con Funciones de Green fuera de equilibrio (NEGF).
- Ecuación de Poisson no lineal con `PETSc` y `MPI`.

**Quantum Code Library (QCL) / libnemo** es una iniciativa para refactorizar NEMO5: pasar de un
ejecutable orientado a una sola simulación a una **librería re-entrante e incrustable**
(*stateless library*) capaz de gestionar varias simulaciones independientes de forma concurrente.

## Recursos de cómputo de Purdue

Usé los clústeres del Rosen Center for Advanced Computing (RCAC):

- **Negishi**: clúster principal de CPU (nodos de 128 núcleos). Allí compilé NEMO5 con la cadena
  Intel 2023 y PETSc 3.5, y lancé trabajos con Slurm (`sinteractive`, `sbatch`).
- **Gilbreth**: cómputo en GPU NVIDIA para código paralelo acelerado con CUDA y cuBLAS.

## Contexto y problema

La desaceleración de la **Ley de Moore** impone barreras físicas severas. A escalas
sub-nanométricas, los efectos cuánticos dominan el transporte de carga y calor, y predecir ese
comportamiento exige simulación HPC de alta precisión. El marco teórico-práctico viene de Pacheco
(*An Introduction to Parallel Programming*) y de Chen y Gupta (*Deep Learning with C++*).

Los cuatro problemas abordados:

1. Ordenamiento paralelo con MPI.
2. Multiplicación de matrices y escalabilidad.
3. Singletons vs. sesiones (RAII) en NEMO5.
4. MLP en CUDA con cuBLAS (Gilbreth).

## 1. Ordenamiento paralelo: Odd-Even Transposition Sort

*Odd-Even Transposition Sort* es una variante de *bubble sort* que hace visible la concurrencia:
compara pares de elementos adyacentes en fases que alternan entre **par** e **impar**. No es el
ordenamiento paralelo más rápido, pero sí el más claro para entender **dependencias**,
**comunicación** y **escalabilidad**.

![Ejemplo gráfico del algoritmo]({{ "/assets/img/tesis/fig1_img.svg" | relative_url }}){: style="width:70%"}
*Ejemplo gráfico del algoritmo.*

- **Fase par:** se comparan las parejas $$(0,1), (2,3), \dots$$
- **Fase impar:** se comparan $$(1,2), (3,4), \dots$$; los extremos sin pareja *descansan*.
- Si una pareja está desordenada, se intercambia (*compare-exchange*).
- Dentro de una fase las parejas no comparten posiciones: **son independientes**.

### ¿Cuántas fases hacen falta?

![Análisis de complejidad computacional]({{ "/assets/img/tesis/fase2_img.png" | relative_url }})
*Análisis de complejidad computacional.*

El menor elemento avanza **a lo sumo una posición** por fase, y puede **perder la primera fase**
si queda sin pareja. Por eso $$n$$ fases garantizan el orden para cualquier entrada (la
demostración formal usa el principio 0-1). El costo serial es de $$n$$ fases por $$\sim n/2$$
comparaciones, es decir, $$\Theta(n^2)$$.

### La versión paralela con MPI

**Problema.** Ordenar listas distribuidas entre $$p$$ procesos evitando condiciones de carrera e
ineficiencias de comunicación (Pacheco, cap. 3).

**Solución algorítmica.** *Odd-Even Transposition Sort* paralelo con MPI. En cada fase (par o
impar), los procesos vecinos intercambian sus sublistas con `MPI_Sendrecv` o
`MPI_Sendrecv_replace` y hacen una fusión local (*merge*): cada uno conserva las claves menores o
mayores según su rango.

|                     | Un elemento por proceso ($$p = n$$) | Bloques ($$p \ll n$$) |
|---------------------|-------------------------------------|-----------------------|
| Datos por proceso   | $$1$$                               | $$n/p$$               |
| Operación por fase  | *compare-exchange*                  | *compare-split*       |
| Número de fases     | $$n$$                               | $$p$$                 |
| Tiempo paralelo     | $$\Theta(n)$$                       | $$\Theta\left(\tfrac{n}{p}\log\tfrac{n}{p}\right) + \Theta(n) + \Theta(n)$$ |

En la versión por bloques, los tres términos son el orden local, los *merges* y la comunicación.
Con bloques, cada proceso mezcla y envía $$\Theta(n)$$ datos en total, **sin importar $$p$$**.

![Odd-Even por bloques con n = 12 y p = 4]({{ "/assets/img/tesis/fig5_img.png" | relative_url }})
*Odd-Even por bloques: $$n = 12$$, $$p = 4$$.*

### Experimento en Negishi

Corrí el algoritmo en el clúster Negishi con $$n = 2^{26} = 67\,108\,864$$ enteros (256 MiB) y
evalué la aceleración y la eficiencia:

$$
S = \frac{T_{\mathrm{serial}}}{T_{\mathrm{paralelo}}}, \qquad E = \frac{S}{p}
$$

![Eficiencia según número de procesos]({{ "/assets/img/tesis/oddeven_03_eficiencia.png" | relative_url }}){: style="width:70%"}
*Eficiencia según número de procesos.*

![Distribución de costos al ordenar]({{ "/assets/img/tesis/oddeven_04_componentes.png" | relative_url }}){: style="width:70%"}
*Distribución de costos al ordenar.*

![Distribución de tiempo]({{ "/assets/img/tesis/exp1_tiempo.png" | relative_url }}){: style="width:80%"}
*Distribución de tiempo.*

## 2. Multiplicación de matrices y escalabilidad en el clúster

![Producto de matrices]({{ "/assets/img/tesis/mat1_producto.svg" | relative_url }}){: style="width:80%"}

En transporte cuántico, casi todo el tiempo de cómputo se va en **productos e inversiones de
matrices**.

### El enfoque clásico (Pacheco): repartir por filas

![Reparto por filas]({{ "/assets/img/tesis/mat3_filas.svg" | relative_url }}){: style="width:60%"}

- Se parte de $$y = Ax$$: cada proceso tiene un bloque de filas de $$A$$ y reúne $$x$$ completo
  con `MPI_Allgather`.
- Para $$C = AB$$, la misma idea obliga a **replicar $$B$$ completa** en cada proceso.
- El cómputo se divide entre $$p$$; los $$\sim n^2$$ datos de $$B$$ no.

### Cómo lo aborda NEMO5: invertir una matriz enorme

![Dispositivo y contactos]({{ "/assets/img/tesis/mat4_dispositivo.svg" | relative_url }}){: style="width:80%"}

$$
G(E) = \left[(E + i\eta)\, I - H - \Sigma_L - \Sigma_R\right]^{-1}
\qquad
T(E) = \mathrm{Tr}\left[\Gamma_R\, G\, \Gamma_L\, G^\dagger\right]
$$

**Ejemplo:** un nanoalambre de Si de $$3\times3\times20$$ nm (Fonseca et al., 2013) tiene
$$\approx 9\,000$$ átomos con 10 orbitales cada uno, así que $$N \approx 9\times10^4$$. La matriz
densa ocupa $$\approx 130$$ GB y requiere $$\sim10^{15}$$–$$10^{16}$$ operaciones **por energía**.

### Recursive Green Function (RGF)

![Matriz tridiagonal por bloques]({{ "/assets/img/tesis/mat5_tridiagonal.svg" | relative_url }}){: style="width:50%"}

**Barrido hacia adelante** (contacto izquierdo incluido):

$$
g_1 = \left(z - H_{11} - \Sigma_L\right)^{-1}
$$

$$
g_i = \left(z - H_{ii} - \tau^\dagger g_{i-1}\, \tau\right)^{-1}, \quad i = 2,\dots,N
$$

**Barrido hacia atrás** (diagonal completa, para la densidad):

$$
G_{ii} = g_i + g_i\, \tau\, G_{i+1,i+1}\, \tau^\dagger g_i
$$

**Esquina** (para la corriente):
diagram.svg
$$
G_{N1} = G_{NN}\, \tau^\dagger g_{N-1}\, \tau^\dagger \cdots\, \tau^\dagger g_1
$$

Aquí $$z = E + i\eta$$ y $$G_{NN}$$ incluye $$\Sigma_R$$. Cada $$g_i$$ es la función de Green
de las rebanadas $$1\ldots i$$, aisladas del resto, y cada paso es un complemento de Schur: la
eliminación gaussiana por bloques de una matriz tridiagonal. Para $$T(E)$$ basta la esquina
$$G_{N1}$$; para la densidad de carga, la diagonal.

### Experimento: RGF vs. inversión densa

RGF se apoya en tres ideas:

1. **Localidad:** cada átomo solo se acopla con sus vecinos, así que la matriz es tridiagonal por
   bloques.
2. **Calcular solo lo necesario:** la diagonal y una columna de bloques de $$G$$, nunca la
   inversa completa.
3. **Bloques pequeños y densos:** $$L$$ inversiones y productos de $$W\times W$$, que caben en
   caché y aprovechan BLAS/LAPACK.

|                 | Costo                   | Memoria                 |
|-----------------|-------------------------|-------------------------|
| Inversión densa | $$O\big((LW)^3\big)$$   | $$O\big((LW)^2\big)$$   |
| **RGF**         | $$O(L\,W^3)$$           | $$O(L\,W^2)$$           |

![Costo de RGF frente a la inversión densa]({{ "/assets/img/tesis/exp2_costo.png" | relative_url }}){: style="width:70%"}
*Costo de RGF frente a la inversión densa.*

La clave no es multiplicar más rápido, sino **multiplicar menos**: se reemplaza una inversión
gigante por muchas pequeñas. Con $$L = 100$$ rebanadas, la diferencia de costo es del orden de
$$L^2 = 10^4$$; en la figura se ve como una pendiente $$\approx 3$$ contra $$\approx 1$$. Si
solo se necesita $$T(E)$$, la memoria baja a $$O(W^2)$$.

## 3. Singletons vs. sesiones (RAII) en NEMO5

NEMO5 se diseñó como un ejecutable por simulación: sus variables globales y *singletons*
corrompen memoria y sobrescriben las bases de materiales al correr simulaciones concurrentes.
Concentré los esfuerzos en mitigar los singletons para explorar la idea de crear la librería QCL
(Quantum Code Library).

![NEMO5 original frente a QCL con sesiones RAII]({{ "/assets/img/tesis/sesiones_arquitectura.svg" | relative_url }})
*Izquierda: todas las simulaciones comparten el estado global. Derecha: cada `Qcl::Session` tiene
su propio estado y sus propios materiales.*

**RAII** (*Resource Acquisition Is Initialization*): cada sesión adquiere sus recursos al crearse
y los libera al destruirse.

### Experimento: dos sesiones concurrentes

Propuse correr dos sesiones concurrentes que crean materiales dentro de un mismo proceso:

![Diagrama del experimento]({{ "/assets/img/tesis/exp_sessions.png" | relative_url }}){: style="width:60%"}
*Diagrama del experimento.*

**Resultados preliminares.** Este primer programa para probar que las instancias están aisladas
dejó los siguientes hallazgos:

![Iceberg de descubrimientos]({{ "/assets/img/tesis/exp_session_findings.png" | relative_url }}){: style="width:60%"}
*Iceberg de descubrimientos.*

### Póster en el Fall Undergraduate Research Expo 2026

Presenté este trabajo como póster en la exposición de investigación de pregrado de Purdue.

![Fotografía del evento]({{ "/assets/img/tesis/expo_2026.jpg" | relative_url }}){: style="width:60%"}
*Fotografía del evento.*

![Póster presentado]({{ "/assets/img/tesis/poster.png" | relative_url }})
*Póster presentado.*

## 4. Presente y futuro de la computación de alto rendimiento

### CPU vs. GPU: dos filosofías de diseño

![Arquitectura CPU frente a GPU]({{ "/assets/img/tesis/gpu_arquitectura.svg" | relative_url }}){: style="width:80%"}

|                       | Nodo Negishi: 2× AMD EPYC 7763 | Gilbreth: 1× NVIDIA A100 40 GB |
|-----------------------|--------------------------------|--------------------------------|
| Unidades de cómputo   | 128 núcleos                    | 108 SM, 6 912 núcleos CUDA (FP32) |
| Frecuencia            | 2.45 GHz base (hasta 3.5)      | 1.41 GHz                       |
| FP64 pico             | ≈ 5 TFLOPS (\*)                | 9.7 TFLOPS (19.5 con núcleos tensoriales) |
| FP32 pico             | ≈ 10 TFLOPS (\*)               | 19.5 TFLOPS (156 en TF32 tensorial) |
| Memoria               | 256 GB DDR4, ≈ 410 GB/s        | 40 GB HBM2, 1 555 GB/s         |
| Caché de último nivel | L3: 2 × 256 MB                 | L2: 40 MB                      |
| Potencia (TDP)        | 2 × 280 W                      | 250–400 W según el formato     |
| Programación          | C++, OpenMP, MPI               | CUDA C++ (kernels), cuBLAS, cuDNN |

(\*) Estimación: núcleos × frecuencia base × 16 FLOP/ciclo (FP64). Coincide con lo medido en el
módulo 2: ≈ 42 GFLOPS por núcleo con BLAS. Datos del A100: ficha técnica de NVIDIA; datos del
EPYC 7763: especificación nominal de AMD.

Una sola GPU tiene **≈ 4× el ancho de banda** y **≈ 2× el pico** de un nodo de 128 núcleos. El
ancho de banda importa tanto como los FLOPS: RGF y muchas simulaciones están limitadas por
memoria, y el pico tensorial TF32 explica por qué la IA migró a GPU.

### El modelo de programación CUDA

![Modelo de programación CUDA]({{ "/assets/img/tesis/gpu_cuda.svg" | relative_url }}){: style="width:80%"}

- La CPU (*host*) reserva memoria en la GPU, copia los datos y **lanza kernels**: funciones que
  ejecutan miles de hilos a la vez.
- Cada hilo calcula su índice (`blockIdx`, `threadIdx`) y procesa su parte de los datos.
- Las copias CPU ↔ GPU son costosas: hay que hacerlas una vez y mantener los datos en la GPU.

### Un perceptrón multicapa (MLP) en CUDA

![Red MLP]({{ "/assets/img/tesis/gpu_mlp.svg" | relative_url }}){: style="width:90%"}

Implementé el perceptrón multicapa del capítulo 5 de Chen y Gupta, con las clases `CudaLayer` y
`CudaMLP`: los productos se hacen con `cublasSgemm` y la ReLU con kernels propios. Otra vez
aparece la multiplicación de matrices, porque entrenar una red es, sobre todo, GEMM, y la
biblioteca (cuBLAS) vuelve a ganarle al código propio. El problema de prueba del libro es XOR;
para medir rendimiento usé una red con capas de ancho variable y un lote de 8 192 muestras.

El experimento corrió en una GPU **NVIDIA A30** del clúster Gilbreth.

![Rendimiento comparado]({{ "/assets/img/tesis/p3_05_tiempo.png" | relative_url }}){: style="width:70%"}
*Rendimiento comparado.*

![Comparativa de aceleración]({{ "/assets/img/tesis/p3_04_speedup.png" | relative_url }}){: style="width:70%"}
*Comparativa de aceleración.*

## Resultados preliminares

1. **Entorno operativo** en Negishi y Gilbreth: compilación de NEMO5 (Intel 2023, PETSc 3.5) y
   entorno CUDA/cuBLAS.
2. **Prototipo de sesiones RAII** para QCL: dos sesiones concurrentes en un mismo proceso; los
   hallazgos definen la hoja de ruta.
3. **Póster** en el Fall Undergraduate Research Expo 2026 de Purdue.
4. **MLP en CUDA** (cap. 5 de Chen y Gupta): implementado con correcciones al código impreso y
   medido en Gilbreth.

Pruebas de regresión aprobadas en Negishi: `bulk_Si`,
`LRAModule_MS_acoustic_form_factor_RGF_solvers` y `NEGF_sanity_wire_new`.

## Conclusiones y trabajo futuro

1. **Paralelizar no basta**: con comunicación en cada fase, Odd-Even se satura en 9×.
2. **El estado global impide la concurrencia**: convertir NEMO5 en librería (QCL) exige
   reemplazar los singletons por sesiones RAII.
3. **MLP en CUDA** (cap. 5 de Chen y Gupta) en una NVIDIA A30: validado contra la CPU; cuBLAS es
   437× más rápido que la CPU con 8 hilos.

**Trabajo futuro:**

- MPI entre GPUs.
- Completar el aislamiento de sesiones en QCL durante la pasantía (hasta febrero de 2027).
- Comparar compiladores (GCC, Intel, Clang) y clústeres (Negishi, Gilbreth).

## Referencias

1. P. S. Pacheco (2011). *An Introduction to Parallel Programming*. Morgan Kaufmann / Elsevier.
2. B. Chen y V. Gupta (2026). *Deep Learning with C++: Design and deploy neural networks using
   CUDA for high-performance AI in C++*. Packt Publishing.
3. J. E. Fonseca, T. Kubis, M. Povolotskyi et al. (2013). Efficient and realistic device modeling
   from atomic detail to the nanoscale: NEMO5, a parallel, multiscale, multiphysics
   nanoelectronics modeling tool. *Journal of Computational Electronics*, 12, 593–608.
4. S. Steiger, M. Povolotskyi, J. H. Park, T. Kubis y G. Klimeck (2011). NEMO5: a parallel
   multiscale nanoelectronics modeling tool. *IEEE Transactions on Nanotechnology*, 10(5),
   1064–1074.
5. Rosen Center for Advanced Computing (2026). *Negishi & Gilbreth Cluster User Guides*. Purdue
   University.