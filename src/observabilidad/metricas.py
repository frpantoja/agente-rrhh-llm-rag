# src/observabilidad/metricas.py
"""
Modulo de metricas de observabilidad para el agente de RRHH.

Implementa metricas de:
- Latencia: tiempo de respuesta total y por herramienta.
- Tasa de exito/error: porcentaje de consultas resueltas sin fallos.
- Uso de herramientas: frecuencia de cada herramienta del agente.
- Consistencia: variabilidad de la longitud de respuesta ante consultas
  similares, como proxy de estabilidad del comportamiento del agente.
- Uso de recursos: tokens de entrada/salida consumidos por el LLM en
  cada consulta (proxy de costo y carga computacional).
- Precision: combina (a) la relevancia semantica de los documentos
  recuperados por la herramienta de consulta (precision de
  recuperacion, capturada automaticamente en cada consulta real) y
  (b) un evaluador basado en casos de prueba con respuesta/herramienta
  esperada conocida (PrecisionEvaluator), para medir si el agente
  elige la herramienta correcta y si la respuesta contiene la
  informacion esperada.

Las metricas se acumulan en memoria durante la sesion y pueden
exportarse a partir de las trazas guardadas en disco para analisis
historico.
"""

import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentMetrics:
    """Acumulador de metricas de una sesion del agente."""

    interacciones: int = 0
    errores: int = 0
    duraciones_ms: List[float] = field(default_factory=list)
    herramientas_usadas: Counter = field(default_factory=Counter)
    duraciones_por_herramienta: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # --- Uso de recursos (tokens del LLM) ---
    tokens_entrada: List[int] = field(default_factory=list)
    tokens_salida: List[int] = field(default_factory=list)
    # --- Precision de recuperacion (scores de similitud semantica) ---
    # Se llena con los scores que devuelve consultar_documentos cada vez
    # que el agente usa esa herramienta (ver src/tools/consulta_tool.py).
    scores_relevancia: List[float] = field(default_factory=list)

    @property
    def tasa_error(self) -> float:
        if not self.interacciones:
            return 0.0
        return round(self.errores / self.interacciones, 3)

    @property
    def tasa_exito(self) -> float:
        return round(1 - self.tasa_error, 3)

    @property
    def latencia_promedio_ms(self) -> float:
        if not self.duraciones_ms:
            return 0.0
        return round(statistics.mean(self.duraciones_ms), 2)

    @property
    def latencia_mediana_ms(self) -> float:
        if not self.duraciones_ms:
            return 0.0
        return round(statistics.median(self.duraciones_ms), 2)

    @property
    def latencia_p95_ms(self) -> float:
        if len(self.duraciones_ms) < 2:
            return self.latencia_promedio_ms
        datos = sorted(self.duraciones_ms)
        idx = min(int(len(datos) * 0.95), len(datos) - 1)
        return round(datos[idx], 2)

    # --- Uso de recursos (tokens) ---

    @property
    def tokens_entrada_totales(self) -> int:
        return sum(self.tokens_entrada)

    @property
    def tokens_salida_totales(self) -> int:
        return sum(self.tokens_salida)

    @property
    def tokens_totales(self) -> int:
        return self.tokens_entrada_totales + self.tokens_salida_totales

    @property
    def tokens_promedio_por_consulta(self) -> float:
        if not self.interacciones:
            return 0.0
        return round(self.tokens_totales / self.interacciones, 1)

    def estimar_costo_usd(self, costo_entrada_por_1k: float = 0.00015,
                           costo_salida_por_1k: float = 0.0006) -> float:
        """
        Estima el costo acumulado en USD a partir de los tokens consumidos.

        Las tarifas por defecto corresponden a un orden de magnitud tipico
        de modelos pequenos tipo gpt-4o-mini (USD por 1000 tokens) y son
        solo referenciales: deben ajustarse a la tarifa real del proveedor
        usado (en este proyecto, la API de GitHub Models).
        """
        costo = (self.tokens_entrada_totales / 1000) * costo_entrada_por_1k
        costo += (self.tokens_salida_totales / 1000) * costo_salida_por_1k
        return round(costo, 5)

    # --- Precision de recuperacion ---

    @property
    def precision_recuperacion_promedio(self) -> Optional[float]:
        """
        Promedio de los scores de relevancia semantica (0 a 1) de los
        documentos efectivamente usados por consultar_documentos en
        cada consulta. Un valor alto indica que el contexto recuperado
        fue, en promedio, muy relevante para la pregunta (baja
        proporcion de "ruido" en el contexto entregado al LLM).
        """
        if not self.scores_relevancia:
            return None
        return round(statistics.mean(self.scores_relevancia), 3)

    def registrar_interaccion(self, duracion_ms: float, exito: bool,
                               herramientas: List[str] = None,
                               tokens_entrada: int = 0,
                               tokens_salida: int = 0,
                               scores_relevancia: List[float] = None) -> None:
        """Registra una consulta completa al agente.

        Args:
            duracion_ms: Duracion total de la consulta.
            exito: True si la consulta termino sin lanzar excepcion.
            herramientas: Lista de herramientas invocadas durante la consulta.
            tokens_entrada: Tokens de entrada (prompt) consumidos por el LLM
                en esta consulta (suma de todos los pasos del agente ReAct).
            tokens_salida: Tokens de salida (completion) consumidos por el LLM.
            scores_relevancia: Scores de similitud semantica de los documentos
                recuperados, si la consulta uso consultar_documentos.
        """
        self.interacciones += 1
        self.duraciones_ms.append(duracion_ms)
        self.tokens_entrada.append(tokens_entrada)
        self.tokens_salida.append(tokens_salida)
        if not exito:
            self.errores += 1

        for herramienta in (herramientas or []):
            self.herramientas_usadas[herramienta] += 1
            self.duraciones_por_herramienta[herramienta].append(duracion_ms)

        if scores_relevancia:
            self.scores_relevancia.extend(scores_relevancia)

    def estado_salud(self) -> str:
        """Clasifica el estado del agente segun la tasa de error."""
        if self.tasa_error > 0.30:
            return "critico"
        if self.tasa_error > 0.10:
            return "degradado"
        return "saludable"

    def resumen(self) -> dict:
        """Retorna un resumen consolidado de todas las metricas."""
        return {
            "interacciones": self.interacciones,
            "tasa_exito": self.tasa_exito,
            "tasa_error": self.tasa_error,
            "estado_salud": self.estado_salud(),
            "latencia_promedio_ms": self.latencia_promedio_ms,
            "latencia_mediana_ms": self.latencia_mediana_ms,
            "latencia_p95_ms": self.latencia_p95_ms,
            "herramientas_usadas": dict(self.herramientas_usadas),
            "tokens_entrada_totales": self.tokens_entrada_totales,
            "tokens_salida_totales": self.tokens_salida_totales,
            "tokens_totales": self.tokens_totales,
            "tokens_promedio_por_consulta": self.tokens_promedio_por_consulta,
            "costo_estimado_usd": self.estimar_costo_usd(),
            "precision_recuperacion_promedio": self.precision_recuperacion_promedio,
        }


class ConsistenciaAnalyzer:
    """
    Mide la consistencia del agente ante consultas del mismo tipo,
    usando la variabilidad en la longitud de la respuesta como proxy.

    Una alta variabilidad en consultas similares puede indicar
    inconsistencia en el comportamiento del agente.
    """

    def __init__(self):
        self._respuestas_por_categoria: Dict[str, List[int]] = defaultdict(list)

    def registrar(self, categoria: str, longitud_respuesta: int) -> None:
        self._respuestas_por_categoria[categoria].append(longitud_respuesta)

    def consistencia_por_categoria(self) -> dict:
        """
        Calcula el coeficiente de variacion (desviacion estandar / media)
        para cada categoria. Valores bajos indican mayor consistencia.
        """
        resultado = {}
        for categoria, longitudes in self._respuestas_por_categoria.items():
            if len(longitudes) < 2:
                resultado[categoria] = {"muestras": len(longitudes), "cv": None}
                continue
            media = statistics.mean(longitudes)
            desv = statistics.stdev(longitudes)
            cv = round(desv / media, 3) if media else 0.0
            resultado[categoria] = {
                "muestras": len(longitudes),
                "media_caracteres": round(media, 1),
                "coeficiente_variacion": cv,
            }
        return resultado


class AnomalyDetector:
    """
    Detecta anomalias de latencia usando z-score sobre una ventana
    deslizante de las ultimas N consultas.
    """

    def __init__(self, window_size: int = 30, z_threshold: float = 2.5):
        self._ventana = deque(maxlen=window_size)
        self._z_threshold = z_threshold

    def procesar(self, duracion_ms: float, trace_id: str = "") -> dict:
        """Evalua si la duracion actual es anomala respecto a la ventana."""
        resultado = {"es_anomalia": False, "z_score": 0.0, "trace_id": trace_id}

        if len(self._ventana) >= 10:
            media = statistics.mean(self._ventana)
            desv = statistics.stdev(self._ventana) if len(self._ventana) > 1 else 0
            if desv > 0:
                z = abs(duracion_ms - media) / desv
                resultado["z_score"] = round(z, 2)
                resultado["es_anomalia"] = z > self._z_threshold

        self._ventana.append(duracion_ms)
        return resultado


@dataclass
class CasoPrueba:
    """
    Caso de prueba con respuesta esperada conocida, usado por
    PrecisionEvaluator para medir la precision del agente de forma
    objetiva (no solo observar su comportamiento, sino contrastarlo
    contra un resultado esperado).

    Attributes:
        pregunta: Consulta exacta usada en la prueba (debe coincidir con
            la que se le hizo al agente, para poder emparejar el caso).
        herramientas_aceptables: Herramientas que se consideran una
            decision correcta para esta pregunta. Lista vacia significa
            que lo correcto es NO usar ninguna herramienta (consulta
            fuera de alcance de RRHH).
        palabras_clave: Palabras o frases que deberian aparecer en una
            respuesta correcta (no sensible a mayusculas/minusculas).
    """
    pregunta: str
    herramientas_aceptables: List[str] = field(default_factory=list)
    palabras_clave: List[str] = field(default_factory=list)


# Casos de prueba por defecto, alineados a los dominios del agente de RRHH.
# IMPORTANTE: reemplaza/ajusta estas preguntas por las 10 consultas reales
# que uses en la seccion "Evidencia de pruebas" del informe, para que el
# resultado de este evaluador sea directamente reutilizable como evidencia.
CASOS_PRUEBA_DEFAULT: List[CasoPrueba] = [
    CasoPrueba(
        pregunta="¿Cuántos días de vacaciones me corresponden?",
        herramientas_aceptables=["consultar_documentos"],
        palabras_clave=["vacaciones", "días"],
    ),
    CasoPrueba(
        pregunta="¿Cómo solicito un permiso administrativo?",
        herramientas_aceptables=["consultar_documentos"],
        palabras_clave=["permiso", "solicitud"],
    ),
    CasoPrueba(
        pregunta="Hazme un resumen del procedimiento de licencias médicas",
        herramientas_aceptables=["generar_resumen", "consultar_documentos"],
        palabras_clave=["licencia", "procedimiento"],
    ),
    CasoPrueba(
        pregunta="¿Cuál es la capital de Francia?",
        herramientas_aceptables=[],
        palabras_clave=["recursos humanos", "rrhh"],
    ),
]


class PrecisionEvaluator:
    """
    Evalua la precision del agente contra un conjunto de casos de
    prueba con resultado esperado conocido.

    A diferencia de las metricas de AgentMetrics (que observan el
    comportamiento del agente sin un "ground truth"), este evaluador
    contrasta cada respuesta real contra lo que se esperaba, lo que
    permite reportar metricas de precision en sentido estricto:

    - precision_herramienta: ¿el agente uso una herramienta aceptable
      para ese tipo de consulta (o correctamente ninguna, si la
      consulta esta fuera de alcance)?
    - precision_contenido: ¿que proporcion de las palabras clave
      esperadas aparecen en la respuesta generada?
    """

    def __init__(self, casos: List[CasoPrueba] = None):
        self._casos: Dict[str, CasoPrueba] = {
            c.pregunta: c for c in (casos if casos is not None else CASOS_PRUEBA_DEFAULT)
        }
        self._resultados: List[dict] = []

    def evaluar(self, pregunta: str, herramientas_usadas: List[str],
                respuesta: str) -> Optional[dict]:
        """
        Compara una respuesta real del agente contra el caso de prueba
        correspondiente (emparejado por texto exacto de la pregunta).

        Returns:
            Diccionario con el detalle de la evaluacion, o None si la
            pregunta no corresponde a ningun caso de prueba registrado.
        """
        caso = self._casos.get(pregunta)
        if caso is None:
            return None

        herramientas_usadas = herramientas_usadas or []
        if caso.herramientas_aceptables:
            herramienta_correcta = any(
                h in caso.herramientas_aceptables for h in herramientas_usadas
            )
        else:
            # Se esperaba que el agente NO usara ninguna herramienta.
            herramienta_correcta = len(herramientas_usadas) == 0

        respuesta_lower = respuesta.lower()
        if caso.palabras_clave:
            encontradas = sum(
                1 for kw in caso.palabras_clave if kw.lower() in respuesta_lower
            )
            precision_contenido = round(encontradas / len(caso.palabras_clave), 3)
        else:
            encontradas = 0
            precision_contenido = None

        resultado = {
            "pregunta": pregunta,
            "herramientas_usadas": herramientas_usadas,
            "herramienta_correcta": herramienta_correcta,
            "palabras_clave_encontradas": encontradas,
            "palabras_clave_totales": len(caso.palabras_clave),
            "precision_contenido": precision_contenido,
        }
        self._resultados.append(resultado)
        return resultado

    def resumen(self) -> dict:
        """Agrega los resultados de todos los casos evaluados hasta ahora."""
        if not self._resultados:
            return {
                "casos_evaluados": 0,
                "precision_herramienta": None,
                "precision_contenido_promedio": None,
            }

        total = len(self._resultados)
        correctos = sum(1 for r in self._resultados if r["herramienta_correcta"])
        contenidos = [
            r["precision_contenido"] for r in self._resultados
            if r["precision_contenido"] is not None
        ]

        return {
            "casos_evaluados": total,
            "precision_herramienta": round(correctos / total, 3),
            "precision_contenido_promedio": (
                round(statistics.mean(contenidos), 3) if contenidos else None
            ),
            "detalle": self._resultados,
        }