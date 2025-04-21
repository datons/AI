from datetime import date as date_type
from typing import List

from pydantic import BaseModel, Field


class DocumentoBOE(BaseModel):
    titulo: str = Field(description='Título del documento publicado en el BOE')
    fecha: date_type = Field(description='Fecha de publicación en el BOE')
    numero: str = Field(description='Número de disposición o referencia oficial')
    departamento: str = Field(description='Departamento u organismo que emite el documento')
    resumen: str = Field(description='Resumen o extracto del contenido del documento')
    url_pdf: str = Field(description='Enlace al PDF oficial del documento')
    url_html: str = Field(description='Enlace al HTML oficial del documento')


class DocumentoBOEList(BaseModel):
    documentos: List[DocumentoBOE]
