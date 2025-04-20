from pydantic import BaseModel, Field
from datetime import date as date_type

class News(BaseModel):
    """
    This is the output format for the news prompt.
    """
    stock: str = Field(description="stock symbol")
    date: date_type = Field(description="date of the news")
    value: float = Field(description="significant accumulated value of the stock")
    title: str = Field(description="title of the news")
    url: str = Field(description="url of the news")
    source: str = Field(description="source of the news")
    explanation: str = Field(description="explanation of the relevance of the news to the stock")

class NewsList(BaseModel):
    """
    This is the output format for the news list prompt.
    """
    news: list[News]