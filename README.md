# Agente Inteligente de RRHH con LLM, RAG y Herramientas

[![CI](https://github.com/frpantoja/agente-rrhh-llm-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/frpantoja/agente-rrhh-llm-rag/actions)

## Descripción

Agente funcional inteligente para consultas internas de Recursos Humanos, desarrollado para la empresa ficticia **Comercial Andina SpA**.

El sistema implementa un agente basado en el patrón **ReAct** (Reasoning + Acting) que integra herramientas de consulta, escritura y razonamiento, junto con un sistema de memoria dual (corto y largo plazo) y planificación adaptativa. Esto le permite decidir autónomamente qué herramienta usar, mantener contexto conversacional y adaptar su comportamiento según el tipo de consulta.

## Arquitectura del Agente

```
                         ┌──────────────────────┐
                         │   Consulta del        │
                         │   trabajador          │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   AGENTE ReAct        │
                         │   (Planificación +    │
                         │    Toma de decisiones) │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
           ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
           │  Herramienta  │ │ Herramienta│ │ Herramienta  │
           │  CONSULTA     │ │ ESCRITURA  │ │ RAZONAMIENTO │
           │  (RAG/FAISS)  │ │ (Resúmenes)│ │ (Análisis)   │
           └────────┬──────┘ └─────┬──────┘ └──────┬───────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │                               │
           ┌────────▼──────┐              ┌────────▼────────┐
           │  MEMORIA       │              │  MEMORIA         │
           │  CORTO PLAZO   │              │  LARGO PLAZO     │
           │  (Historial    │              │  (RAG: FAISS +   │
           │   conversación)│              │   Embeddings)    │
           └───────────────┘              └─────────────────┘
```

### Componentes del Sistema

| Componente | Tecnología | Propósito |
|---|---|---|
| Agente | LangChain Agents (ReAct) | Orquestación y toma de decisiones |
| Tool: Consulta | FAISS + Embeddings | Búsqueda semántica en documentos |
| Tool: Escritura | LLM (gpt-4o-mini) | Generación de resúmenes y documentos |
| Tool: Razonamiento | LLM + multi-fuente | Análisis de situaciones complejas |
| Memoria corto plazo | ConversationBufferWindowMemory | Historial conversacional (últimas 5) |
| Memoria largo plazo | FAISS + text-embedding-3-small | Base vectorial persistente |
| Embeddings | text-embedding-3-small (Azure) | Representación vectorial |
| LLM | gpt-4o-mini (Azure) | Generación de respuestas |
| CI/CD | GitHub Actions | Tests automáticos |

## Estructura del Proyecto

```
asistente-rrhh-llm-rag/
├── app.py                          # Interfaz de consola del agente
├── dashboard.py                    # Dashboard de observabilidad (Streamlit)
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuración centralizada
├── src/
│   ├── __init__.py
│   ├── agente.py                   # Agente funcional (orquestador)
│   ├── memoria.py                  # Memoria corto y largo plazo
│   ├── cargar_documentos.py        # Carga de documentos con metadatos
│   ├── crear_vectores.py           # Pipeline de indexación FAISS
│   ├── prompts.py                  # Templates de prompts
│   ├── rag_pipeline.py             # Pipeline RAG (base)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── consulta_tool.py        # Herramienta de consulta documental
│   │   ├── escritura_tool.py       # Herramienta de escritura/resúmenes
│   │   └── razonamiento_tool.py    # Herramienta de razonamiento normativo
│   └── observabilidad/
│       ├── __init__.py
│       ├── trazas.py               # Trace_id, spans y logging estructurado
│       └── metricas.py             # Latencia, recursos, consistencia, anomalías, precisión
├── tests/
│   ├── __init__.py
│   ├── test_agente.py
│   ├── test_cargar_documentos.py
│   ├── test_memoria.py
│   ├── test_metricas.py
│   ├── test_prompts.py
│   ├── test_rag_pipeline.py
│   ├── test_tools.py
│   └── test_trazas.py
├── data/
│   ├── internos/                   # Documentos corporativos simulados
│   └── externos/                   # Normativa laboral de referencia
├── logs/                           # Trazas JSON (generadas en ejecución, no versionadas)
├── evidencias/                     # Capturas de pruebas
├── .github/workflows/ci.yml       # Pipeline CI
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Observabilidad y Trazabilidad

El sistema registra cada consulta con un **trace_id** único que conecta todos los pasos de su procesamiento (invocación del agente, llamadas a herramientas, generación de respuesta). Los eventos se guardan en `logs/trazas.jsonl` en formato JSON estructurado.

### Métricas implementadas

| Métrica | Descripción | Módulo |
|---|---|---|
| Latencia | Tiempo total por consulta, promedio, mediana y p95 | `AgentMetrics` |
| Tasa de éxito/error | Porcentaje de consultas resueltas sin fallos | `AgentMetrics` |
| Uso de herramientas | Frecuencia de cada herramienta del agente | `AgentMetrics` |
| Uso de recursos | Tokens de entrada/salida consumidos por el LLM por consulta y acumulados, con estimación de costo | `AgentMetrics` |
| Consistencia | Variabilidad de longitud de respuesta por tipo de consulta | `ConsistenciaAnalyzer` |
| Anomalías | Detección de picos de latencia vía z-score sobre ventana deslizante | `AnomalyDetector` |
| Precisión de recuperación | Relevancia semántica promedio de los documentos efectivamente usados por `consultar_documentos` | `AgentMetrics` |
| Precisión (casos de prueba) | % de selección correcta de herramienta y de palabras clave esperadas presentes en la respuesta, contra un set de preguntas con resultado esperado conocido | `PrecisionEvaluator` |

### Minimización de datos

Las consultas de los trabajadores se registran como **hash SHA-256**, no como texto plano, en los archivos de traza. Esto reduce el riesgo de exponer información personal en los logs, en línea con principios de minimización de datos de la Ley 21.719 sobre protección de datos personales en Chile.

### Dashboard de monitoreo

```bash
streamlit run dashboard.py
```

El dashboard muestra: latencia por consulta en el tiempo, uso de recursos (tokens consumidos por el LLM por consulta y acumulados), precisión de recuperación (relevancia semántica promedio del contexto), distribución de uso de herramientas, tasa de error y estado de salud del agente (saludable/degradado/crítico), anomalías detectadas, y un listado filtrable de las consultas más lentas.

## Herramientas del Agente

### 1. Herramienta de Consulta (`consultar_documentos`)
Realiza búsqueda semántica en la base vectorial FAISS para recuperar fragmentos relevantes de los documentos de RRHH. Implementa filtrado por umbral de similitud (0.3) para evitar resultados irrelevantes.

**Cuándo se usa**: Preguntas directas sobre políticas, procedimientos o normativas.

### 2. Herramienta de Escritura (`generar_resumen`)
Genera documentos estructurados como resúmenes de procedimientos, correos formales de solicitud y explicaciones paso a paso de trámites.

**Cuándo se usa**: Cuando el trabajador necesita un documento formal o un resumen escrito.

### 3. Herramienta de Razonamiento (`analizar_situacion_laboral`)
Analiza situaciones laborales complejas que requieren cruzar información de múltiples documentos (internos y externos) y aplicar lógica normativa.

**Cuándo se usa**: Casos del tipo "¿qué pasa si...?" o situaciones con múltiples condiciones.

## Sistema de Memoria

### Memoria de Corto Plazo

El sistema implementa tres estrategias de memoria de corto plazo, seleccionables por configuracion (`MEMORY_TYPE` en `.env`):

| Estrategia | Clase | Comportamiento | Caso de uso |
|---|---|---|---|
| **Buffer** | `MemoriaBuffer` | Guarda todo el historial sin limite | Conversaciones cortas donde se necesita todo el contexto |
| **Window** | `MemoriaWindow` | Ventana deslizante de las ultimas K interacciones | Conversaciones moderadas, balancea contexto y tokens |
| **Summary** | `MemoriaSummary` | Resume la conversacion usando el LLM cada 2 turnos | Conversaciones largas donde importa el contexto general |

Por defecto se usa **Summary Memory**, que genera un resumen progresivo de la conversacion mediante el LLM. Esto permite mantener el contexto general sin exceder el limite de tokens del modelo, incluso en conversaciones prolongadas.

### Memoria de Largo Plazo
- **Tecnología**: FAISS + embeddings (`text-embedding-3-small`).
- **Persistencia**: Guardada en disco (`faiss_index/`).
- **Propósito**: Almacenar y recuperar semánticamente los documentos de RRHH.
- **Documentos**: 5 internos + 3 externos, divididos en chunks de 300 caracteres.

## Planificación y Toma de Decisiones

El agente sigue un esquema de planificación con orden estricto de evaluación y prioridades definidas:

```
PASO 1: ¿Es tema de RRHH?
        |
        No --> Responder directamente, fin.
        |
        Si
        v
PASO 2: Clasificación por prioridad
        |
        a) ¿Requiere documento/resumen?  --> generar_resumen
        |  (prioridad sobre consulta simple)
        |
        b) ¿Caso complejo / multi-norma? --> analizar_situacion_laboral
        |
        c) Caso simple                   --> consultar_documentos
        v
PASO 3: ¿La consulta depende del historial?
        |
        Si --> Reformular antes de buscar
        |      (ej: "¿cómo las solicito?" → "cómo solicitar vacaciones")
        v
PASO 4: ¿Se necesitan varias herramientas en secuencia?
        |
        Si --> Ejecutar en orden y combinar resultados
```

Este esquema asegura que, ante una consulta que podría calzar con más de una herramienta (por ejemplo, pedir un correo formal sobre vacaciones, que es a la vez documento y consulta simple), el agente sabe cuál tiene prioridad: generar el documento, que a su vez requiere consultar primero.

### Resolución de contexto en preguntas de seguimiento

Una limitación frecuente en agentes con RAG es que preguntas de seguimiento cortas ("¿y cómo las solicito?") pierden precisión en la búsqueda semántica porque no contienen suficiente información por sí solas. El agente reformula estas consultas incorporando el tema de la conversación previa antes de invocar `consultar_documentos`, asegurando que la recuperación semántica siga siendo precisa incluso en conversaciones prolongadas con múltiples turnos.

### Ejemplos de toma de decisiones

| Consulta | Decisión del agente | Herramienta |
|---|---|---|
| "¿Cuántos días de vacaciones tengo?" | Consulta directa → buscar en documentos | `consultar_documentos` |
| "Hazme un correo para pedir vacaciones" | Necesita generar documento formal (prioridad sobre consulta simple) | `generar_resumen` + `consultar_documentos` |
| "Si estoy con licencia, ¿puedo pedir vacaciones?" | Caso complejo, cruzar normativas | `analizar_situacion_laboral` |
| "¿Cuál es la capital de Francia?" | Fuera de alcance → respuesta directa | Ninguna |
| "¿Y cómo las solicito?" (seguimiento) | Reformula con el tema previo antes de buscar | `consultar_documentos` |

## Requisitos Previos

- Python 3.10 o superior
- Git
- Token personal de GitHub con permisos de lectura de modelos

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/frpantoja/agente-rrhh-llm-rag.git
cd agente-rrhh-llm-rag

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu GITHUB_TOKEN personal
```

## Uso

### 1. Crear la base vectorial (primera vez o si cambian los documentos)

```bash
python -m src.crear_vectores
```

### 2. Ejecutar el agente

```bash
python app.py
```

### Comandos disponibles en la consola

| Comando | Acción |
|---|---|
| Escribir una consulta | El agente la procesa y responde |
| `memoria` | Muestra el estado de la memoria conversacional |
| `limpiar` | Limpia la memoria de corto plazo |
| `metricas` | Muestra el resumen de métricas de la sesión actual (latencia, tasa de éxito, tokens consumidos, costo estimado y precisión de recuperación) |
| `consistencia` | Muestra el análisis de consistencia por tipo de consulta |
| `precision` | Evalúa las consultas hechas hasta ahora contra los casos de prueba con resultado esperado conocido (`CASOS_PRUEBA_DEFAULT`) |
| `salir` | Termina el programa |

### Ejemplos de consultas para probar

**Consultas simples** (usa herramienta de consulta):
- ¿Cuántos días de vacaciones me corresponden?
- ¿Cómo solicito un permiso administrativo?
- ¿Qué beneficios ofrece la empresa?

**Solicitudes de escritura** (usa herramienta de escritura):
- Hazme un resumen del procedimiento de licencias médicas
- Redacta un correo para solicitar vacaciones

**Casos complejos** (usa herramienta de razonamiento):
- Si estoy con licencia médica, ¿puedo pedir vacaciones al mismo tiempo?
- ¿Qué diferencia hay entre lo que dice el reglamento interno y el Código del Trabajo sobre permisos?

**Seguimiento con memoria** (prueba de continuidad):
- Primero: "¿Qué dice la política de vacaciones?"
- Luego: "¿Y cómo las solicito?"
- Luego: "¿Con cuánta anticipación?"

**Fuera de alcance** (prueba de guardrails):
- ¿Cuál es la capital de Francia?
- ¿Puedo trabajar desde casa?

### 3. Ver el dashboard de observabilidad

Tras realizar varias consultas con `app.py` (esto genera trazas en `logs/trazas.jsonl`), abre el dashboard en otra terminal:

```bash
streamlit run dashboard.py
```

### 4. Ejecutar tests

```bash
python -m pytest tests/ -v
```

## Decisiones Técnicas

### ¿Por qué LangChain Agents con ReAct?
El patrón ReAct permite al agente razonar explícitamente antes de actuar. A diferencia de un pipeline lineal (siempre RAG → LLM), el agente evalúa la consulta y decide el mejor camino. Esto es más flexible y escalable que un sistema de reglas fijas.

### ¿Por qué separar en 3 herramientas?
Cada herramienta tiene un propósito claro y distinto. La separación permite al agente combinarlas según necesite y facilita agregar nuevas herramientas en el futuro sin modificar el agente.

### ¿Por qué tres estrategias de memoria?
Cada estrategia tiene ventajas y limitaciones distintas. Buffer es simple pero costosa en tokens. Window balancea contexto reciente con eficiencia. Summary consume tokens constantes sin importar la longitud de la conversacion, pero pierde detalles especificos. Implementar las tres permite elegir la mas adecuada segun el escenario y demuestra comprension de los trade-offs involucrados.

### ¿Por qué chunks de 300 caracteres?
Los documentos de RRHH son cortos (~650 chars promedio). Chunks de 300 con overlap de 80 capturan secciones individuales con contexto suficiente, mejorando la precisión del retrieval respecto a chunks más grandes.

## Tecnologías y Frameworks

- **Python 3.10+**: Lenguaje principal
- **LangChain**: Framework de agentes, herramientas y memoria
- **FAISS**: Base vectorial para búsqueda de similitud
- **OpenAI (via Azure)**: Modelos de embedding y LLM
- **GitHub Actions**: CI/CD
- **pytest**: Testing

## Uso Ético de IA

Este proyecto fue desarrollado con apoyo de inteligencia artificial para mejorar redacción, organización y orientación técnica. El análisis del caso, diseño de la solución, arquitectura y validación fueron realizados por el equipo.

## Referencias

- LangChain. (2024). *LangChain Documentation*. https://python.langchain.com/docs/
- LangChain. (2024). *Agents*. https://python.langchain.com/docs/concepts/agents/
- LangChain. (2024). *Memory*. https://python.langchain.com/docs/concepts/memory/
- Facebook AI Research. (2024). *FAISS: A Library for Efficient Similarity Search*. https://github.com/facebookresearch/faiss
- Yao, S., Zhao, J., Yu, D., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*. https://arxiv.org/abs/2210.03629
- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. https://arxiv.org/abs/2005.11401

## Autoría

Proyecto académico para la asignatura **Ingeniería de Soluciones con Inteligencia Artificial**.