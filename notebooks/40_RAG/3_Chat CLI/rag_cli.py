#!/usr/bin/env python3

import os
import argparse
import datetime
import hashlib
import json
import uuid
import getpass
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
workspace_env_path = Path("/workspace/.env")
if workspace_env_path.exists():
    load_dotenv(dotenv_path=workspace_env_path)
    print(f"Loaded environment variables from {workspace_env_path}")
else:
    local_env_path = Path(".env")
    if local_env_path.exists():
        load_dotenv(dotenv_path=local_env_path)
        print(f"Loaded environment variables from {local_env_path}")

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers.base import BaseRetriever
from langchain.schema import Document

from vectordb.bocyl import get_vectordb_bocyl


class SafeRetriever(BaseRetriever):
    """A wrapper around a retriever that ensures the query is properly processed."""

    def __init__(self, retriever):
        self.retriever = retriever

    def _get_relevant_documents(self, query, **kwargs):
        """Process the query and then call the underlying retriever."""
        try:
            # Ensure query is a string
            if isinstance(query, dict) and "question" in query:
                processed_query = query["question"]
            elif isinstance(query, str):
                processed_query = query
            else:
                processed_query = str(query)

            print(f"Searching with query: {processed_query}")

            # Call the underlying retriever
            return self.retriever._get_relevant_documents(processed_query, **kwargs)
        except Exception as e:
            import traceback

            print(f"Error in retriever: {str(e)}")
            print(traceback.format_exc())
            # Return an empty list of documents in case of error
            return [Document(page_content=f"Error retrieving documents: {str(e)}")]


class RAGChatCLI:
    def __init__(self, conversation_id: Optional[str] = None):
        # Setup directories
        self.conversations_dir = "conversations"
        os.makedirs(self.conversations_dir, exist_ok=True)

        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            api_key = getpass.getpass("Introduce tu OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key

        # Initialize vectordb and RAG components
        self.vectordb = get_vectordb_bocyl()
        base_retriever = self.vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        # Wrap the retriever with our safe retriever
        self.retriever = SafeRetriever(base_retriever)

        # Create prompt template
        template = """Answer the question based only on the following context:

{context}

Question: {question}

Take into account the conversation history:
{history}

Answer in Spanish. If the answer cannot be found in the context, say "No encuentro información sobre esto en los documentos proporcionados."
"""
        self.prompt = ChatPromptTemplate.from_template(template)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o"
        )  # Changed to gpt-4o instead of chatgpt-4o-latest

        # Format docs function - fixed to properly handle document format
        self.format_docs = self.safe_format_docs

        # Create or load conversation
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.history = []
        self.load_conversation()

        # Create RAG chain
        self.rag_chain = self.create_rag_chain()

    def safe_format_docs(self, docs):
        """Safely format documents, handling different possible formats."""
        formatted_docs = []
        for doc in docs:
            try:
                # Handle standard document objects with page_content attribute
                if hasattr(doc, "page_content"):
                    formatted_docs.append(doc.page_content)
                # Handle dictionary objects with 'page_content' key
                elif isinstance(doc, dict) and "page_content" in doc:
                    formatted_docs.append(doc["page_content"])
                # Handle plain string documents
                elif isinstance(doc, str):
                    formatted_docs.append(doc)
                # Handle other cases
                else:
                    formatted_docs.append(str(doc))
            except Exception as e:
                formatted_docs.append(f"Error formatting document: {str(e)}")

        return "\n\n".join(formatted_docs)

    def create_rag_chain(self):
        return (
            {
                "context": self.retriever | self.format_docs,
                "question": lambda x: x["question"],
                "history": lambda x: self.format_history(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_history(self) -> str:
        if not self.history:
            return "No hay historial de conversación previo."

        formatted_history = []
        for entry in self.history:
            formatted_history.append(f"Usuario: {entry['question']}")
            formatted_history.append(f"Asistente: {entry['response']}")

        return "\n".join(formatted_history)

    def load_conversation(self):
        conversation_file = os.path.join(
            self.conversations_dir, f"{self.conversation_id}.json"
        )
        if os.path.exists(conversation_file):
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)
                self.history = conversation_data.get("messages", [])
                print(f"Cargando conversación: {self.conversation_id}")
                if self.history:
                    print("\nHistorial de conversación:")
                    for i, entry in enumerate(self.history):
                        print(f"[{i + 1}] Usuario: {entry['question']}")
                        print(f"    Asistente: {entry['response']}\n")

    def save_conversation(self):
        conversation_file = os.path.join(
            self.conversations_dir, f"{self.conversation_id}.json"
        )
        conversation_data = {
            "id": self.conversation_id,
            "created_at": datetime.datetime.now().isoformat(),
            "messages": self.history,
        }
        with open(conversation_file, "w") as f:
            json.dump(conversation_data, f, indent=2)

        # Also save the latest response as a markdown file in outputs directory
        if self.history:
            latest = self.history[-1]
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            response_hash = hashlib.md5(latest["response"].encode()).hexdigest()
            os.makedirs("outputs", exist_ok=True)

            filename = f"outputs/{current_datetime}_{response_hash}.md"
            with open(filename, "w") as f:
                f.write(latest["response"])

    def ask(self, question: str) -> str:
        try:
            response = self.rag_chain.invoke({"question": question})

            # Add to history
            self.history.append(
                {
                    "question": question,
                    "response": response,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            # Save conversation
            self.save_conversation()

            return response
        except Exception as e:
            import traceback

            print(f"Error procesando la pregunta: {str(e)}")
            print(traceback.format_exc())
            return (
                f"Lo siento, hubo un error al procesar tu pregunta. Detalles: {str(e)}"
            )

    def list_conversations(self) -> List[str]:
        conversations = []
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith(".json"):
                conversation_id = filename.replace(".json", "")
                conversation_file = os.path.join(self.conversations_dir, filename)

                with open(conversation_file, "r") as f:
                    data = json.load(f)
                    first_message = (
                        data.get("messages", [{}])[0].get(
                            "question", "Nueva conversación"
                        )
                        if data.get("messages")
                        else "Nueva conversación"
                    )
                    created_at = data.get("created_at", "")
                    conversations.append(
                        {
                            "id": conversation_id,
                            "first_message": first_message[:50]
                            + ("..." if len(first_message) > 50 else ""),
                            "created_at": created_at,
                            "message_count": len(data.get("messages", [])),
                        }
                    )

        return sorted(
            conversations, key=lambda x: x.get("created_at", ""), reverse=True
        )

    def run_cli(self):
        print(f"Conversación activa: {self.conversation_id}")
        print(
            "Escribe 'salir' para terminar, 'nueva' para crear una nueva conversación,"
        )
        print("o 'listar' para ver todas las conversaciones disponibles.")

        while True:
            question = input("\n> ")

            if question.lower() == "salir":
                break
            elif question.lower() == "nueva":
                self.conversation_id = str(uuid.uuid4())
                self.history = []
                print(f"Nueva conversación creada: {self.conversation_id}")
            elif question.lower() == "listar":
                conversations = self.list_conversations()
                if not conversations:
                    print("No hay conversaciones guardadas.")
                else:
                    print("\nConversaciones disponibles:")
                    for i, conv in enumerate(conversations):
                        print(f"[{i + 1}] ID: {conv['id']}")
                        print(f"    Primer mensaje: {conv['first_message']}")
                        print(f"    Creada: {conv['created_at']}")
                        print(f"    Mensajes: {conv['message_count']}\n")

                    selection = input(
                        "Selecciona una conversación (número) o presiona Enter para continuar: "
                    )
                    if selection.isdigit() and 1 <= int(selection) <= len(
                        conversations
                    ):
                        selected_conv = conversations[int(selection) - 1]
                        self.conversation_id = selected_conv["id"]
                        self.history = []
                        self.load_conversation()
            else:
                try:
                    response = self.ask(question)
                    print(f"\n{response}")
                except Exception as e:
                    print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG Chat CLI")
    parser.add_argument("--conversation", "-c", help="ID de conversación existente")
    parser.add_argument("--api-key", "-k", help="OpenAI API key")
    args = parser.parse_args()

    # Set API key from command line if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    cli = RAGChatCLI(conversation_id=args.conversation)
    cli.run_cli()


if __name__ == "__main__":
    main()
