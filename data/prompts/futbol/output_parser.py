from datetime import date as tipo_fecha
from typing import List

from pydantic import BaseModel, Field


class Partido(BaseModel):
    liga: str = Field(description='Nombre de la liga de fútbol')
    jornada: int = Field(description='Número de la jornada en la temporada')
    fecha: tipo_fecha = Field(description='Fecha en la que se juega el partido')
    equipo_local: str = Field(description='Nombre del equipo local')
    equipo_visitante: str = Field(description='Nombre del equipo visitante')
    goles_local: int = Field(description='Goles anotados por el equipo local')
    goles_visitante: int = Field(description='Goles anotados por el equipo visitante')
    estadio: str = Field(description='Nombre del estadio donde se juega el partido')
    url: str = Field(description='URL con más información del partido')


class ListaPartidos(BaseModel):
    partidos: List[Partido]
