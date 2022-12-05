### Question Answering AI

Streamlit GUI that receives any text input from a PDF, Image OCR, Wikipedia Page, URL, or plain text, & utilizes the AllenAi/longformer-larger-4096 model to answer a question based on that information.

Given there is a max length to the model, if the total text passed as the input is larger than the max size, it will chunk it into appropriate sizes by sentence, and will run the model iteratively.

## How to use

1. Select input type
2. Click analyse
3. Ask any question in the prompt
