from fastapi import FastAPI
from data_in import PostQuestion

app = FastAPI(
    title="Question Image Upload"
)

@app.post("/question_image/")
def create_question_image(question: PostQuestion):
    return {"question": question}