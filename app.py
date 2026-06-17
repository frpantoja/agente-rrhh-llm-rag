"""
Interfaz de consola del Agente de RRHH.

Implementa la interfaz de usuario para interactuar con el agente funcional.
El agente mantiene memoria conversacional entre preguntas y decide
qué herramientas usar para responder. Cada consulta queda registrada
con un trace_id para fines de observabilidad y trazabilidad.

Comandos especiales:
- 'salir' / 'exit': Terminar el programa.
- 'memoria': Ver el estado de la memoria del agente.
- 'limpiar': Limpiar la memoria conversacional.
- 'metricas': Ver el resumen de métricas de la sesión actual.
- 'consistencia': Ver el análisis de consistencia por tipo de consulta.
"""

import logging

from config.settings import LOG_LEVEL, LOG_FORMAT
from src.agente import AgenteRRHH

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 60)
    print("  Agente Inteligente de RRHH - Comercial Andina SpA")
    print("  Escribe tu consulta o 'salir' para terminar.")
    print("  Comandos: 'memoria' | 'limpiar' | 'metricas' | 'consistencia' | 'salir'")
    print("=" * 60 + "\n")

    try:
        agente = AgenteRRHH()
    except ValueError as e:
        print(f"\nError de configuracion: {e}")
        print("Revisa el archivo .env y asegurate de tener un GITHUB_TOKEN valido.")
        return

    while True:
        try:
            pregunta = input("Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPrograma finalizado.")
            break

        if not pregunta:
            continue

        if pregunta.lower() in ("salir", "exit", "quit"):
            print("Programa finalizado.")
            break

        if pregunta.lower() == "memoria":
            print(f"\n{agente.obtener_estado_memoria()}")
            continue

        if pregunta.lower() == "limpiar":
            agente.limpiar_memoria()
            print("\nMemoria conversacional limpiada.\n")
            continue

        if pregunta.lower() == "metricas":
            metricas = agente.obtener_metricas()
            print("\nMetricas de la sesion:")
            for clave, valor in metricas.items():
                print(f"  {clave}: {valor}")
            print()
            continue

        if pregunta.lower() == "consistencia":
            consistencia = agente.obtener_consistencia()
            print("\nConsistencia por tipo de consulta:")
            for categoria, datos in consistencia.items():
                print(f"  {categoria}: {datos}")
            print()
            continue

        try:
            resultado = agente.consultar(pregunta)

            print(f"\nRespuesta:")
            print(resultado.respuesta)

            if resultado.herramientas_usadas:
                print(f"\nHerramientas utilizadas: {', '.join(resultado.herramientas_usadas)}")

            print(f"\ntrace_id: {resultado.trace_id} | duracion: {resultado.duracion_ms}ms")
            print("\n" + "-" * 60 + "\n")

        except Exception as e:
            logger.error("Error al procesar consulta: %s", e, exc_info=True)
            print(f"\nOcurrio un error: {e}")
            print("Intenta reformular tu consulta.\n")


if __name__ == "__main__":
    main()
