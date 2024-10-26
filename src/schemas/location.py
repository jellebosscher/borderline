from pydantic import BaseModel, Field


class Location(BaseModel):

    name: str = Field(...)

    x: float = Field(...)

    y: float = Field(...)
