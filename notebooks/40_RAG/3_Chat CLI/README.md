# RAG Chat CLI

Una interfaz de línea de comandos para chatear con un sistema RAG (Retrieval Augmented Generation) que consulta documentos de la Junta de Castilla y León.

## Características

- Conversa con el LLM usando documentos como contexto
- Guarda las conversaciones automáticamente
- Permite cargar conversaciones anteriores
- Almacena las respuestas en archivos Markdown
- Mantiene historial de conversaciones

## Configuración de API Key

La aplicación necesita una API key de OpenAI. Puedes configurarla de las siguientes formas:

1. En un archivo `.env` en el directorio raíz del workspace (`/workspace/.env`) o en el directorio local:
   ```
   OPENAI_API_KEY=tu_api_key
   ```
   La aplicación buscará primero en `/workspace/.env` y luego en el directorio local `.env`.

2. Como variable de entorno:
   ```bash
   export OPENAI_API_KEY=tu_api_key
   ```

3. Usando el argumento de línea de comandos:
   ```bash
   python rag_cli.py --api-key=tu_api_key
   ```
   o
   ```bash
   python rag_cli.py -k tu_api_key
   ```

4. Manualmente cuando se te solicite durante la ejecución del programa.

## Uso

### Iniciar una nueva conversación

```bash
python rag_cli.py
```

### Continuar una conversación existente

```bash
python rag_cli.py --conversation ID_DE_CONVERSACION
```

o

```bash
python rag_cli.py -c ID_DE_CONVERSACION
```

### Comandos durante la ejecución

- `salir` - Terminar la sesión actual
- `nueva` - Crear una nueva conversación
- `listar` - Ver todas las conversaciones disponibles

## Estructura de directorios

- `conversations/` - Almacena los archivos JSON de cada conversación
- `outputs/` - Contiene las respuestas individuales en formato Markdown

## Requisitos

Este script utiliza los mismos componentes que el notebook RAG:

- El vectordb configurado desde `modules.vectordb`
- Langchain
- OpenAI API
- python-dotenv (para cargar el archivo .env)

## Ejemplo de uso

```
$ python rag_cli.py
Loaded environment variables from /workspace/.env
Conversación activa: 9b3a7e8c-f123-4567-89ab-cdef01234567
Escribe 'salir' para terminar, 'nueva' para crear una nueva conversación,
o 'listar' para ver todas las conversaciones disponibles.

> ¿Qué documentos hay sobre condicionalidad reforzada?

En los documentos proporcionados, se mencionan varios documentos relacionados con la condicionalidad reforzada:
...

> listar

Conversaciones disponibles:
[1] ID: 9b3a7e8c-f123-4567-89ab-cdef01234567
    Primer mensaje: ¿Qué documentos hay sobre condicionalidad reforzada?
    Creada: 2023-04-25T15:30:22.123456
    Mensajes: 1

Selecciona una conversación (número) o presiona Enter para continuar: 

> salir 