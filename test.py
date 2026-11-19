from pydantic import BaseModel

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float

print(WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.95))