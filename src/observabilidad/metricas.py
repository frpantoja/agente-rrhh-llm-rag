"""
Modulo de metricas de observabilidad para el agente de RRHH.

Implementa metricas de:
- Latencia: tiempo de respuesta total y por herramienta.
- Tasa de exito/error: porcentaje de consultas resueltas sin fallos.
- Uso de herramientas: frecuencia de cada herramienta del agente.
- Consistencia: variabilidad de la longitud de respuesta ante consultas
  similares, como proxy de estabilidad del comportamiento del agente.

Las metricas se acumulan en memoria durante la sesion y pueden
exportarse a partir de las trazas guardadas en disco para analisis
historico.
"""

import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List


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

    def registrar_interaccion(self, duracion_ms: float, exito: bool,
                               herramientas: List[str] = None) -> None:
        """Registra una consulta completa al agente."""
        self.interacciones += 1
        self.duraciones_ms.append(duracion_ms)
        if not exito:
            self.errores += 1

        for herramienta in (herramientas or []):
            self.herramientas_usadas[herramienta] += 1
            self.duraciones_por_herramienta[herramienta].append(duracion_ms)

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
