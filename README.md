# ACS2024-2_GRUPO4
PROYECTO FINAL -  PENDULO INVERTIDO CON CONTROL INTELIGENTE
1. Implementa un controlador inteligente utilizando algoritmos de aprendizaje por refuerzo clásico (Q-learning o SARSA). Diseña un agente que aprenda a estabilizar el péndulo en el entorno simulado CartPole-v1 de Gym.
2. Familiarízate con los parámetros del entorno CartPole-v1 (longitud del péndulo, masa del carro y fuerza máxima aplicable). Analiza cómo estos afectan el comportamiento del sistema y la capacidad del agente para aprender una política de control efectiva.
3. Diseña una función de recompensa que fomente mantener el péndulo en posición vertical, penalizando el movimiento excesivo del carro y las caídas. Simula el sistema para
comprobar cómo esta estructura de recompensa influye en el aprendizaje del agente.
4. Entrena el agente en el entorno durante un número específico de episodios, y evalúa cuánto tiempo le toma aprender una política estable. Propongan estrategias para acelerar el proceso de convergencia, como ajustes en la tasa de aprendizaje o exploración. Por ejemplo, ajustar el balance entre exploración y explotación.
5. Realiza un análisis gráfico y escrito que responda las siguientes preguntas:
    a. ¿Cómo afecta la longitud del péndulo o la fuerza máxima aplicable en el carro a la capacidad del agente de estabilizar el sistema?<br>
    b. ¿Cómo influye la estructura de la función de recompensa en el desempeño del agente? Por ejemplo, ¿qué efecto tiene aumentar o disminuir las penalizaciones por movimientos excesivos del carro o la caída del péndulo?<br>
   c. ¿Cuánto tiempo tarda el agente en aprender una política estable? Analiza cómo la tasa de aprendizaje, el factor de descuento (γ) y la política de exploración afectan el tiempo de convergencia.<br>
   d. ¿Cómo responde el agente a estados iniciales desfavorables, como un ángulo del péndulo cercano al límite de caída o a una posición extrema del carro?<br>
   e. Genera gráficos que representen la evolución de la recompensa acumulada por episodio durante el entrenamiento

## Requisitos previos

- [Python 3.11](https://www.python.org/downloads/) o una versión compatible.
- [pip](https://pip.pypa.io/en/stable/installation/) para la gestión de paquetes.

## Configuración del entorno

Sigue estos pasos para configurar el entorno de desarrollo:

1. **Clona este repositorio**:
   ```bash
   git clone https://github.com/BryanAnth/ACS2024-2_GRUPO4.git
   cd ACS2024-2_GRUPO4
   ```

2. **Crea un entorno virtual**:
   Ejecuta el siguiente comando para crear un entorno virtual en la carpeta `venv`:
   ```bash
   python -m venv venv
   ```

3. **Activa el entorno virtual**:
   - En **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - En **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Instala las dependencias**:
   Con el entorno virtual activo, instala las dependencias necesarias ejecutando:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
```bash
python main.py
```

