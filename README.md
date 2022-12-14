### Question Answering AI

Streamlit GUI that receives any text input from a PDF, Image OCR, Wikipedia Page, URL, or plain text, to answer a question based on that information.

Given there is a max length to the model, if the total text passed as the input is larger than the max size, it will chunk it into appropriate sizes by sentence, and will run the model iteratively.

![Example](https://github.com/iamramizk/question-answering-ai/blob/main/assets/qa-ai-showcase.gif)

## How to use

1. Select input type
2. Add input source
3. Click analyse
4. Ask any question in the prompt

## Running the app

### Hosted via streamlit cloud

[Live link](https://iamramizk-question-answering-ai-app-q9gnoi.streamlit.app/)

### Installing locally

```
git clone https://github.com/iamramizk/question-answering-ai
cd question-answering-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
