"""Tests para el módulo de memoria del agente."""

from src.memoria import (
    MemoriaConversacional,
    crear_memoria_corto_plazo,
    obtener_resumen_memoria,
    MEMORY_WINDOW_SIZE,
)


def test_crear_memoria_retorna_instancia():
    memoria = crear_memoria_corto_plazo()
    assert memoria is not None
    assert isinstance(memoria, MemoriaConversacional)


def test_memoria_vacia_al_inicio():
    memoria = crear_memoria_corto_plazo()
    assert len(memoria.obtener_mensajes()) == 0


def test_agregar_interaccion():
    memoria = crear_memoria_corto_plazo()
    memoria.agregar_interaccion("Hola", "Hola, ¿en qué puedo ayudarte?")
    assert len(memoria.obtener_mensajes()) == 2


def test_ventana_deslizante():
    memoria = MemoriaConversacional(window_size=2)
    memoria.agregar_interaccion("Pregunta 1", "Respuesta 1")
    memoria.agregar_interaccion("Pregunta 2", "Respuesta 2")
    memoria.agregar_interaccion("Pregunta 3", "Respuesta 3")
    # Ventana de 2 = 4 mensajes máximo
    assert len(memoria.obtener_mensajes()) == 4


def test_resumen_memoria_vacia():
    memoria = crear_memoria_corto_plazo()
    resumen = obtener_resumen_memoria(memoria)
    assert "vacía" in resumen.lower()


def test_resumen_memoria_con_datos():
    memoria = crear_memoria_corto_plazo()
    memoria.agregar_interaccion("¿Cuántos días de vacaciones tengo?", "Según la política...")
    resumen = obtener_resumen_memoria(memoria)
    assert "2 mensajes" in resumen


def test_limpiar_memoria():
    memoria = crear_memoria_corto_plazo()
    memoria.agregar_interaccion("Hola", "Hola")
    memoria.limpiar()
    assert len(memoria.obtener_mensajes()) == 0
