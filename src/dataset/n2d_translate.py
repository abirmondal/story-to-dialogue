"""
n2d_translate.py

Translate narrative-to-dialogue text and character names using the
`ai4bharat/indictrans2-en-indic-1B` NMT model via Hugging Face Transformers.
"""

import json
from typing import List, Dict, Optional

import torch
from tqdm.auto import tqdm
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class N2DTranslate:
    """Handle translation using ai4bharat/indictrans2-en-indic-1B.

        Notes:
        - This class uses `IndicProcessor` from IndicTransToolkit.
        - Provide language codes (e.g. `eng_Latn`, `ben_Beng`) via
            `source_lang_code` / `target_lang_code`. Defaults are English -> Bengali.
    """

    def __init__(
        self,
        translation_model_name: str = "ai4bharat/indictrans2-en-indic-1B",
        device: Optional[str] = None,
        source_lang_code: str = "eng_Latn",
        target_lang_code: str = "ben_Beng",
        use_fp16: bool = True,
        attn_implementation: Optional[str] = None,
    ) -> None:
        """Initialize model, tokenizer and Indic processor.

        Args:
            translation_model_name: HF model repo id.
            device: device string, default autodetected.
            source_lang_code: source language code used by IndicProcessor.
            target_lang_code: target language code used by IndicProcessor.
            use_fp16: whether to load model weights in float16 (recommended on CUDA).
            attn_implementation: optional attention implementation for performance.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.source_lang_code = source_lang_code
        self.target_lang_code = target_lang_code

        # Tokenizer and model loading
        self.tokenizer = AutoTokenizer.from_pretrained(
            translation_model_name, trust_remote_code=True
        )

        torch_dtype = torch.float16 if use_fp16 else torch.float32

        model_kwargs = dict(
            **{
                "trust_remote_code": True,
                "dtype": torch_dtype,
            }
        )

        # Optionally include attention implementation if provided
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            translation_model_name, **model_kwargs
        ).to(self.device)

        # Some IndicTrans2 + transformers versions fail when decoder cache is enabled.
        self.model.config.use_cache = False

        self.ip = IndicProcessor(inference=True)

    def _translate_batch(self, inputs: List[str], max_length: int = 512) -> List[str]:
        if not inputs:
            return []

        # Pre-process inputs to add language tags and clean text
        processed_inputs = self.ip.preprocess_batch(
            inputs,
            src_lang=self.source_lang_code,
            tgt_lang=self.target_lang_code,
        )

        # Tokenize inputs with truncation and padding
        tokenized = self.tokenizer(
            processed_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Generate translations with no_grad for efficiency
        with torch.inference_mode():
            generated_tokens = self.model.generate(
                **tokenized,
                min_length=0,
                max_length=max_length,
                num_beams=5,
                use_cache=False,
                num_return_sequences=1,
            )

        # Decode and post-process translations
        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # Post-process removes the tags and cleans up extra spaces
        translations = self.ip.postprocess_batch(decoded, lang=self.target_lang_code)

        return [t.strip() for t in translations]

    def _translate_name_batch(self, names: List[str]) -> Dict[str, str]:
        """
        Translate a list of character names directly using IndicTrans2.
        """
        if not names:
            return {}

        translated_names = self._translate_batch(names, max_length=128)

        if not translated_names:
            return {n: n for n in names}  # Fallback to original name if failed

        sanitized_translations = [
            t.strip().replace(".", "") for t in translated_names
        ]

        if len(sanitized_translations) < len(names):
            print(
                f"Warning: Name mismatch. Expected {len(names)}, got {len(sanitized_translations)}")
            while len(sanitized_translations) < len(names):
                sanitized_translations.append(names[len(sanitized_translations)])

        return dict(zip(names, sanitized_translations))

    def _save_name_mapping(self, name_mapping: Dict[str, str], filename: str = "name_mapping.json") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, ensure_ascii=False, indent=4)

    def translate_names(self, names: List[str], batch_size: int = 128, save_mapping: bool = True, filename: str = "name_mapping.json") -> Dict[str, str]:
        """Translate a list of character names in batches and optionally save mapping.
        """
        full_mapping: Dict[str, str] = {}
        for i in tqdm(range(0, len(names), batch_size), desc="Translating names"):
            batch_names = names[i : i + batch_size]
            batch_mapping = self._translate_name_batch(batch_names)
            full_mapping.update(batch_mapping)

        if save_mapping:
            self._save_name_mapping(full_mapping, filename)

        return full_mapping

    def translate_text(self, text: str, max_length: int = 512) -> str:
        """Translate the input text and return the translated string.
        """
        translations = self._translate_batch([text], max_length=max_length)
        return translations[0] if translations else ""
    
    def translate_texts(self, texts: List[str], batch_size: int = 16, max_length: int = 512) -> List[str]:
        """Translate a list of texts in batches and return the list of translated strings.
        """
        translated_texts: List[str] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating texts"):
            batch_texts = texts[i : i + batch_size]
            batch_translations = self._translate_batch(batch_texts, max_length=max_length)
            translated_texts.extend(batch_translations)
        return translated_texts
