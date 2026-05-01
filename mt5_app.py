from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

MT5_DIR = "mt5_model"
TRANSLATOR_MODEL_NAME = "facebook/nllb-200-distilled-600M"

SUPPORTED_LANGS = ["english", "spanish", "french", "russian", "portuguese"]

NLLB_LANG_CODES = {
    "english": "eng_Latn",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "russian": "rus_Cyrl",
    "portuguese": "por_Latn",
}

MT5_LANG_PROMPTS = {
    "english": "English",
    "spanish": "Spanish",
    "french": "French",
    "russian": "Russian",
    "portuguese": "Portuguese",
}

mt5_tokenizer = AutoTokenizer.from_pretrained(MT5_DIR, use_fast=False)
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(MT5_DIR).to(device)
mt5_model.eval()

translator_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL_NAME)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL_NAME).to(device)
translator_model.eval()


class SummarizeTranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    summary_max_new_tokens: int = 64
    translation_max_new_tokens: int = 128
    num_beams: int = 4


def validate_lang(lang: str) -> None:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Language must be one of {SUPPORTED_LANGS}")


def generate_mt5_summary(
    text: str,
    src_lang: str,
    max_new_tokens: int = 64,
    num_beams: int = 4,
) -> str:
    validate_lang(src_lang)

    prompt_lang = MT5_LANG_PROMPTS[src_lang]

    prompt = f"summarize in {prompt_lang}: {text}"

    inputs = mt5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = mt5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=mt5_model.config.decoder_start_token_id,
            eos_token_id=mt5_model.config.eos_token_id,
            pad_token_id=mt5_model.config.pad_token_id,
        )

    return mt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def translate_text(
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> str:
    validate_lang(src_lang)
    validate_lang(tgt_lang)

    src_code = NLLB_LANG_CODES[src_lang]
    tgt_code = NLLB_LANG_CODES[tgt_lang]

    translator_tokenizer.src_lang = src_code

    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = translator_model.generate(
            **inputs,
            forced_bos_token_id=translator_tokenizer.convert_tokens_to_ids(tgt_code),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    return translator_tokenizer.decode(output_ids[0], skip_special_tokens=True)


@app.get("/")
def root():
    return {"message": "mT5 API is running"}


@app.post("/summarize-translate")
def summarize_translate(req: SummarizeTranslateRequest):
    source_summary = generate_mt5_summary(
        text=req.text,
        src_lang=req.src_lang,
        max_new_tokens=req.summary_max_new_tokens,
        num_beams=req.num_beams,
    )

    translated_summary = translate_text(
        text=source_summary,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        max_new_tokens=req.translation_max_new_tokens,
        num_beams=req.num_beams,
    )

    return {
        "model": "mt5",
        "source_summary": source_summary,
        "translated_summary": translated_summary,
    }