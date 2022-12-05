import streamlit as st
import os

# import torch
from allennlp.predictors.predictor import Predictor

# from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from rapidfuzz import fuzz
from build.ocr import text_from_images
from build.wiki import get_wiki_page
from build.url import text_from_url
from build.pdf import text_from_pdf
from build.chunks import corpus_to_chucks


os.environ["TOKENIZERS_PARALLELISM"] = "true"


st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide".
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="Question Answering AI",  # String or None.
    page_icon="🗨",  # String, anything supported by st.image, or None.
)

with open("build/style.css", "r") as css:
    st.markdown(f"<style>{css.read().strip()}</style>", unsafe_allow_html=True)


with st.sidebar:
    st.image("assets/qa-logo.png", width=200)
    st.subheader(
        "This AI will learn from any text you give it, and can answer questions from the information."
    )
    options = ["Images", "Text", "Wikipedia", "URL", "PDF"]
    input_select = st.radio("Select Input", options)
    st.write("")
    sidebar_footer = st.empty()

# Initialise global session state variables
if "text" not in st.session_state:
    st.session_state["text"] = ""
if "wordcount" not in st.session_state:
    st.session_state["wordcount"] = 0


# @st.cache(hash_funcs={LongformerTokenizerFast: hash}, suppress_st_warning=True)
# def load_model_tokenizer():
#     """Loads the model and tokenizer into chache"""
#     model_name = "allenai/bidaf-elmo"  # smaller model for stremalit hosting
#     # model_name = "allenai/longformer-large-4096-finetuned-triviaqa"
#     tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
#     model = LongformerForQuestionAnswering.from_pretrained(model_name)
#     return model, tokenizer


# model, tokenizer = load_model_tokenizer()


# def get_answer(question: str, text: str) -> str:
#     """Runs the model to get an answer to a question
#     Corpus provided as text"""
#     encoding = tokenizer(question, text, return_tensors="pt")
#     input_ids = encoding["input_ids"]

#     attention_mask = encoding["attention_mask"]

#     outputs = model(input_ids, attention_mask=attention_mask)
#     start_logits = outputs.start_logits
#     end_logits = outputs.end_logits
#     all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

#     answer_tokens = all_tokens[
#         torch.argmax(start_logits) : torch.argmax(end_logits) + 1
#     ]
#     answer = tokenizer.decode(
#         tokenizer.convert_tokens_to_ids(answer_tokens)
#     )  # remove space prepending space token

#     return answer.strip().capitalize()


@st.cache(allow_output_mutation=True)
# @st.cache(hash_funcs={Predictor: hash}, suppress_st_warning=True)
def load_predictor():
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz"
    )
    return predictor


predictor = load_predictor()


def get_answer(question: str, text: str) -> list:
    """Runs the model to get an answer to a question,
    returns [answer: str, score: float]"""
    output = predictor.predict(passage=text, question=question)
    answer = output["best_span_str"]
    score = max(output["span_end_probs"])
    return [answer, score]


if input_select:  # ["Images", "Text", "Wikipedia", "URL", "PDF"]
    st.title(input_select)

    if input_select == "Images":
        images = st.sidebar.file_uploader(
            "Images to analyse text from", accept_multiple_files=True
        )
        if images:
            analyse = st.sidebar.button("Analyse Images")

            if analyse:
                st.session_state["text"] = text_from_images(images)
                st.session_state["wordcount"] = len(st.session_state["text"].split())

    elif input_select == "Text":
        text_input = st.sidebar.text_area("Text to analyse")
        if text_input:
            load_text = st.sidebar.button("Analyse Text")
            if load_text:
                st.session_state["text"] = text_input.strip()
                st.session_state["wordcount"] = len(st.session_state["text"].split())

    elif input_select == "Wikipedia":
        query = st.sidebar.text_input(
            "Keyword to analyse wikipedia content",
            placeholder="Eg. Spirulina or Donald Trump",
        )

        if query:
            scrape = st.sidebar.button("Analyse Wiki")
            if scrape:
                st.session_state["text"] = get_wiki_page(query)
                st.session_state["wordcount"] = len(st.session_state["text"].split())

    elif input_select == "URL":
        input_url = st.sidebar.text_input(
            "Website URL to analyse content from",
            placeholder="https://google.com/...",
        )
        if input_url:
            scrape = st.sidebar.button("Analyse URL")
            if scrape:
                st.session_state["text"] = text_from_url(input_url)
                st.session_state["wordcount"] = len(st.session_state["text"].split())

    elif input_select == "PDF":
        input_pdf = st.sidebar.file_uploader("PDF File")
        if input_pdf:
            analyse = st.sidebar.button("Analyse Document")
            if analyse:
                st.session_state["text"] = text_from_pdf(input_pdf)
                st.session_state["wordcount"] = len(st.session_state["text"].split())


def is_similar(text: str, previous_answers: list, ratio: int = 90) -> bool:
    """Checks weather an answer is similar to list of previous answers"""
    if not previous_answers:
        return False
    for previous_answer in previous_answers:
        if fuzz.ratio(text, previous_answer) > ratio:
            return True
    return False


if len(st.session_state["text"]) > 2:
    st.sidebar.write(" ")
    st.sidebar.caption(f"Total Words: {st.session_state['wordcount']}")
    question = st.text_input("Enter Question")
    if question:
        if not question.endswith("?"):
            question = question.strip() + "?"
        text_chunks = corpus_to_chucks(st.session_state["text"], chunk_max_words=400)

        st.markdown("***")
        previous_answers = []
        print_count = 1
        for chunk_idx, text_chunk in enumerate(text_chunks, start=1):
            with st.spinner(f"Working on it... {chunk_idx}/{len(text_chunks)}"):
                answer, score = get_answer(question, text_chunk)

            # If answer is not too similar to previous answer, show answer
            if not is_similar(answer, previous_answers, ratio=95) and answer:
                st.caption(f"Answer {print_count} - Confidence: {round(score, 2)}")
                st.write(answer)
                st.write("")
                print_count += 1
                previous_answers.append(answer)
