from pydantic import BaseModel, Field

class News(BaseModel):
    stock: str = Field(description="símbolo de la acción")
    date: str = Field(description="fecha de la noticia")
    value: float = Field(description="valor acumulado significativo de la acción")
    title: str = Field(description="título de la noticia")
    url: str = Field(description="url de la noticia")
    source: str = Field(description="fuente de la noticia")
    explanation: str = Field(description="explicación de la relevancia de la noticia para la acción")

class NewsList(BaseModel):
    news: list[News] = Field(description="lista de News")