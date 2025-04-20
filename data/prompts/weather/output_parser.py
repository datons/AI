from pydantic import BaseModel, Field
from datetime import date as date_type
from typing import List

class Weather(BaseModel):
    """
    This is the output format for the weather prompt.
    """
    
    location: str = Field(description="location of the weather")
    date: date_type = Field(description="date of the weather")
    description: str = Field(description="description of the weather")
    temperature_min: float = Field(description="minimum temperature of the weather")
    temperature_max: float = Field(description="maximum temperature of the weather")
    temperature_avg: float = Field(description="average temperature of the weather")
    url: str = Field(description="url of the weather")

class WeatherList(BaseModel):
    """
    This is the output format for the weather list prompt.
    """
    weathers: List[Weather]