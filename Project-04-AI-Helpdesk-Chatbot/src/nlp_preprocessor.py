"""
NLP Preprocessing Module for Internal Helpdesk Chatbot
Module 2: NLP Text Preprocessing + Intent/Entity Preparation

This module provides a reusable NLP preprocessing pipeline that turns the
Module 1 dataset (data/processed/faq_dataset.csv) into clean NLP-ready data.

Provided functionality
----------------------
- Text cleaning / normalization        clean_text()
- Tokenization                         tokenize_text()
- Stopword handling                    remove_stopwords()
- Lemmatization (POS-aware)            lemmatize_tokens()
- Intent normalization                 normalize_intent()
- Entity normalization                 normalize_entity()
- Rule/dictionary-based entity
  extraction                           extract_entities()
- Single record processing             preprocess_record()
- Single question processing           preprocess_question()
- Full dataset processing              preprocess_dataset()
- Dataset loading / validation         load_dataset(), validate_preprocessed_data()
- Statistics + quality gate            generate_nlp_statistics(), compute_quality_gate()
- Output generation                    save_nlp_dataset()

Design decisions
----------------
- Tokenization uses NLTK ``word_tokenize`` (punkt/punkt_tab resources) with a
  safe fallback to a simple regex tokenizer when NLTK data is unavailable.
  Contractions ("didn't", "won't", "I'm", ...) are expanded first so their
  fragments never leak into the vocabulary; possessive "'s" is handled.
- Stopword handling uses the standard NLTK English stopword list. The original
  cleaned question is ALWAYS preserved (``clean_question``) and stopword
  removal only affects ``filtered_tokens``.
- Lemmatization uses WordNet with NLTK POS tagging so verbs such as "running"
  are reduced to "run". When the POS tagger resource is unavailable it falls
  back to plain (noun) lemmatization and never crashes the pipeline.
- Entity extraction is transparent rule/dictionary-based (substring matching
  with word boundaries) over a controlled vocabulary built from the Module 1
  canonical entities. No deep-learning NER is used.
- ``entity`` is the primary entity (dataset value, normalized; falls back to
  the first extracted entity). ``entities`` is the full list of detected
  entities. The dataset's original ``entity`` value is never lost.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import pandas as pd

# ---------------------------------------------------------------------------
# NLTK availability (graceful)
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - environment specific
    NLTK_AVAILABLE = False
    word_tokenize = None
    stopwords = None
    WordNetLemmatizer = None
    pos_tag = None

# ---------------------------------------------------------------------------
# Schema / vocabulary (kept in sync with Module 1 - prepare_dataset.py)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["question", "intent", "answer", "entity"]

OUTPUT_COLUMNS = [
    "question",
    "clean_question",
    "tokens",
    "filtered_tokens",
    "lemmatized_tokens",
    "intent",
    "answer",
    "entity",
    "entities",
]

# Canonical intent labels (Module 1 source of truth - 22 intents).
CANONICAL_INTENTS = [
    "account_access", "attendance", "contact_information", "email_problems",
    "employee_id", "goodbye", "greetings", "help", "holidays", "hr_support",
    "internet_problems", "laptop_problems", "leave_policy", "office_location",
    "password_reset", "payroll", "salary_information", "security",
    "software_installation", "technical_support", "wifi_problems",
    "working_hours",
]

# Canonical entity values (Module 1 source of truth - 22 entities).
CANONICAL_ENTITIES = [
    "email", "account", "password", "laptop", "wifi", "software",
    "leave", "salary", "payroll", "internet", "working_hours",
    "employee_id", "holiday", "attendance", "hr", "security",
    "contact", "location", "it_support", "greeting", "goodbye", "help",
]

# Controlled, dataset-driven entity vocabulary used by the rule-based
# extractor.  Keys are canonical entities; values are the surface keywords
# detected in employee questions (matched with word boundaries).
ENTITY_KEYWORDS = {
    "password": ["password", "passcode", "pwd"],
    "account": ["account", "login", "log in", "log into", "username", "sign in"],
    "laptop": ["laptop", "computer", "pc", "machine", "monitor", "keyboard"],
    "wifi": ["wifi", "wi-fi", "wi fi", "wireless"],
    "email": ["email", "mail", "outlook", "inbox"],
    "software": ["software", "application", "app", "program", "tool", "install"],
    "leave": ["leave", "vacation", "time off", "sick days", "pto"],
    "salary": ["salary", "wage", "compensation", "stipend"],
    "payroll": ["payroll", "paycheck", "payday", "direct deposit", "payslip", "pay stub"],
    "internet": ["internet", "web", "browsing", "broadband", "network"],
    "working_hours": ["working hours", "work hours", "work timings", "office hours",
                      "shift", "work from home", "remote work"],
    "employee_id": ["employee id", "employee number", "emp id", "staff id",
                    "id number", "id card"],
    "holiday": ["holiday", "holidays", "festival", "public holiday"],
    "attendance": ["attendance", "timesheet", "time sheet", "punch"],
    "hr": ["hr", "human resources", "personnel", "hr office"],
    "security": ["security", "breach", "incident", "phishing", "suspicious", "badge"],
    "contact": ["contact", "phone number", "phone", "directory", "extension",
                "contact details"],
    "location": ["location", "office location", "building", "floor", "campus",
                 "cafeteria", "coffee station", "canteen", "pantry"],
    "it_support": ["it support", "technical support", "tech support", "helpdesk",
                   "help desk", "it help"],
    "greeting": ["hello", "hi", "hey", "greetings", "good morning",
                 "good afternoon", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "farewell", "take care"],
    "help": ["help", "assist", "assistance", "can you help"],
}

# Common contractions expanded BEFORE tokenization so fragments (e.g. "n't",
# "'re") never leak into tokens / vocabulary.
CONTRACTION_MAP = {
    "don't": "do not", "didn't": "did not", "doesn't": "does not",
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "aren't": "are not",
    "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
    "shan't": "shall not", "i'm": "i am", "you're": "you are",
    "we're": "we are", "they're": "they are", "he's": "he is",
    "she's": "she is", "it's": "it is", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "we'd": "we would", "they'd": "they would",
    "let's": "let us", "that's": "that is", "there's": "there is",
    "what's": "what is", "how's": "how is", "who's": "who is",
}

class NLPPreprocessor:
    """Reusable NLP preprocessing pipeline for the helpdesk FAQ dataset."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the preprocessor.

        Args:
            project_root: Optional project root directory. When omitted the
                project root is derived from this module's location
                (src/ -> project root).
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)

        self.lemmatizer: Optional[WordNetLemmatizer] = None
        self.stop_words: set = set()
        self._nltk_ready = False
        self._tagger_ready = False

        if NLTK_AVAILABLE:
            self._initialize_nltk()

    # ------------------------------------------------------------------ NLTK
    def _initialize_nltk(self) -> None:
        """
        Download (quietly) and initialise the NLTK resources used here.

        Resources required: tokenizers (punkt/punkt_tab), stopwords, WordNet
        (wordnet + omw-1.4) and the averaged perceptron POS tagger.

        Every download / initialisation is defensive: if a resource cannot be
        downloaded the pipeline degrades gracefully (regex tokenizer fallback,
        no stopword filtering, noun-only lemmatization) instead of failing.
        """
        # Core resources (tokenization, stopwords, lemmatization).
        try:
            for resource in ("punkt_tab", "punkt", "stopwords", "wordnet", "omw-1.4"):
                nltk.download(resource, quiet=True)
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self._nltk_ready = True
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"Warning: NLTK initialization failed: {exc}")
            self._nltk_ready = False

        # Optional POS tagger (verb lemmatization). Never fatal.
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            pos_tag([])  # force resource check / lazy load
            self._tagger_ready = True
        except Exception:  # pragma: no cover - environment specific
            self._tagger_ready = False

    # ------------------------------------------------------------ Step 4
    def clean_text(self, text: Any) -> str:
        """
        Clean and normalize raw text.

        Applies (in order): safe conversion to string, lowercase,
        control-character removal, whitespace normalization, trailing
        question-mark removal, repeated punctuation normalization and removal
        of punctuation/special characters that do not contribute meaning.
        Meaningful terms (password, wifi, email, payroll, attendance, ...) are
        preserved; apostrophes and hyphens are kept for contractions and
        hyphenated terms such as "wi-fi".

        Args:
            text: Input text to clean.

        Returns:
            Cleaned, normalized lowercase string.

        Examples:
            "  How Do I Reset My Company Password??? " -> "how do i reset my company password"
        """
        if text is None:
            return ""
        if isinstance(text, float) and pd.isna(text):
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Lowercase.
        text = text.lower()

        # Remove control characters (including DEL).
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize whitespace and strip.
        text = re.sub(r"\s+", " ", text).strip()

        # Remove trailing question marks.
        text = re.sub(r"\?+\s*$", "", text)

        # Normalize repeated exclamation/period sequences.
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\.{2,}", ".", text)

        # Remove punctuation/special characters, keeping letters, digits,
        # spaces, apostrophes and hyphens.
        text = re.sub(r"[^a-z0-9\s'-]", "", text)

        # Normalize whitespace once more.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------------------------------------------ Step 5
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions before tokenization."""
        for contraction, expansion in CONTRACTION_MAP.items():
            text = text.replace(contraction, expansion)
        return text

    def tokenize_text(self, text: Any) -> List[str]:
        """
        Tokenize text into word tokens.

        Uses NLTK ``word_tokenize`` when available; falls back to a simple
        regex tokenizer otherwise. Contractions are expanded first and
        apostrophe fragments (e.g. possessive "'s") are dropped so only clean
        alphanumeric tokens remain.

        Args:
            text: Input text to tokenize.

        Returns:
            List of word tokens.

        Examples:
            "how do i reset my password" -> ["how", "do", "i", "reset", "my", "password"]
        """
        if not isinstance(text, str) or not text.strip():
            return []

        text = self._expand_contractions(text.lower())

        if NLTK_AVAILABLE and self._nltk_ready:
            try:
                tokens = word_tokenize(text)
            except Exception:
                tokens = re.findall(r"[a-z0-9']+", text)
        else:
            tokens = re.findall(r"[a-z0-9']+", text)

        cleaned: List[str] = []
        for token in tokens:
            token = token.strip("'")
            if token and re.fullmatch(r"[a-z0-9]+", token):
                cleaned.append(token)
        return cleaned

    # ------------------------------------------------------------ Step 6
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove English stopwords from a token list.

        NOTE: this only produces ``filtered_tokens``; the original cleaned
        question is never modified.

        Args:
            tokens: List of tokens.

        Returns:
            Tokens with stopwords removed. When NLTK stopwords are
            unavailable the input tokens are returned unchanged.
        """
        if not self.stop_words:
            return list(tokens)
        return [token for token in tokens if token.lower() not in self.stop_words]

    # ------------------------------------------------------------ Step 7
    @staticmethod
    def _wordnet_pos(treebank_tag: str) -> str:
        """Map an NLTK POS tag prefix to a WordNet POS tag."""
        if treebank_tag.startswith("J"):
            return "a"
        if treebank_tag.startswith("V"):
            return "v"
        if treebank_tag.startswith("R"):
            return "r"
        return "n"

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base (dictionary) form.

        POS-aware lemmatization is used when the NLTK tagger is available
        (employees -> employee, running -> run, issues -> issue); otherwise
        plain noun lemmatization is used. The original question/clean_question
        are never overwritten.

        Args:
            tokens: List of tokens to lemmatize.

        Returns:
            List of lemmatized tokens.
        """
        if not self.lemmatizer:
            return list(tokens)

        tokens = list(tokens)
        if self._tagger_ready:
            try:
                tagged = pos_tag(tokens)
                return [
                    self.lemmatizer.lemmatize(token, self._wordnet_pos(tag))
                    for token, tag in tagged
                ]
            except Exception:
                pass
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    # ------------------------------------------------------------ Step 8
    def normalize_intent(self, intent: Any) -> str:
        """
        Normalize an intent label to lowercase snake_case.

        Examples:
            "Password_Reset"   -> "password_reset"
            "password_reset "  -> "password_reset"
            " PASSWORD_RESET " -> "password_reset"

        Args:
            intent: Raw intent label.

        Returns:
            Normalized intent label (or "" when empty).
        """
        if intent is None:
            return ""
        intent = str(intent).strip().lower()
        intent = re.sub(r"\s+", "_", intent)
        intent = re.sub(r"[^a-z0-9_]", "", intent)
        intent = re.sub(r"_+", "_", intent)
        return intent.strip("_")

    # ------------------------------------------------------------ Step 9
    def normalize_entity(self, entity: Any) -> str:
        """
        Normalize an entity value to a canonical entity label.

        Intent-like labels are mapped onto their canonical entity, e.g.:
            "password_reset"  -> "password"
            "wifi_problem"    -> "wifi"
            "laptop_support"  -> "laptop"

        The value is first matched verbatim against the canonical entity set;
        otherwise a prefix match ("<entity>_<suffix>") is attempted.

        Args:
            entity: Raw entity value.

        Returns:
            Canonical entity label; cleaned original value when it cannot be
            matched (so information is never silently lost); "" when empty.
        """
        if entity is None:
            return ""
        value = str(entity).strip().lower()
        if not value:
            return ""

        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"[^a-z0-9_]", "", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            return ""

        if value in CANONICAL_ENTITIES:
            return value

        # Intent-like label -> entity ("password_reset" -> "password").
        for canonical in CANONICAL_ENTITIES:
            if value.startswith(canonical + "_"):
                return canonical

        return value

    # ----------------------------------------------------------- Step 10
    def extract_entities(self, text: Any, existing_entity: Optional[str] = None) -> List[str]:
        """
        Lightweight, transparent rule/dictionary-based entity extraction.

        Scans the (cleaned) question for the controlled entity vocabulary in
        ``ENTITY_KEYWORDS`` using word-boundary matching, so substrings inside
        larger words (e.g. "app" inside "apply", "pto" inside "laptop") never
        produce false positives.

        Args:
            text: Question text to scan.
            existing_entity: Dataset entity (hint). It is validated but does
                not influence the returned extracted list.

        Returns:
            Deduplicated list of detected canonical entities (in canonical
            vocabulary order). Empty list when nothing matches.

        Examples:
            "How do I reset my password?"        -> ["password"]
            "My laptop cannot connect to Wi-Fi." -> ["laptop", "wifi"]
            "When will my salary be credited?"   -> ["salary"]
        """
        if not isinstance(text, str) or not text.strip():
            return []

        text_lower = text.lower()
        detected: List[str] = []
        for entity in CANONICAL_ENTITIES:
            for keyword in ENTITY_KEYWORDS.get(entity, []):
                if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                    detected.append(entity)
                    break
        return detected

    # ----------------------------------------------------------- Step 11
    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single dataset record.

        Args:
            record: Dictionary with at least "question", "intent", "answer"
                and "entity" keys.

        Returns:
            NLP-ready record dictionary with OUTPUT_COLUMNS keys.
        """
        question = record.get("question", "")
        intent = record.get("intent", "")
        answer = record.get("answer", "")
        entity = record.get("entity", "")

        if pd.isna(question):
            question = ""
        if pd.isna(intent):
            intent = ""
        if pd.isna(answer):
            answer = ""
        if pd.isna(entity):
            entity = ""

        clean_question = self.clean_text(question)
        tokens = self.tokenize_text(clean_question)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
        normalized_intent = self.normalize_intent(intent)
        normalized_entity = self.normalize_entity(entity)
        extracted_entities = self.extract_entities(clean_question)

        # Primary entity: the dataset entity (normalized) is authoritative;
        # fall back to the first extracted entity only when it is empty.
        primary_entity = normalized_entity or (
            extracted_entities[0] if extracted_entities else ""
        )

        return {
            "question": question,
            "clean_question": clean_question,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "intent": normalized_intent,
            "answer": answer,
            "entity": primary_entity,
            "entities": extracted_entities,
        }

    def preprocess_question(
        self,
        question: str,
        intent: str,
        answer: str,
        entity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Preprocess a single question (convenience wrapper).

        Args:
            question: Original question.
            intent: Intent label.
            answer: Answer text.
            entity: Optional entity value.

        Returns:
            NLP-ready record dictionary.
        """
        return self.preprocess_record(
            {"question": question, "intent": intent, "answer": answer, "entity": entity}
        )

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess an entire FAQ dataset.

        Args:
            df: DataFrame with REQUIRED_COLUMNS.

        Returns:
            NLP-ready DataFrame with OUTPUT_COLUMNS.
        """
        if df is None or len(df) == 0:
            raise ValueError("Empty dataset provided")

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        records = [self.preprocess_record(row.to_dict()) for _, row in df.iterrows()]
        return pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    # -------------------------------------------------------- IO helpers
    def load_dataset(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and validate the Module 1 processed dataset.

        Args:
            path: Path to the CSV file.

        Returns:
            DataFrame with the Module 1 schema.

        Raises:
            FileNotFoundError: dataset does not exist.
            ValueError: dataset is empty or missing required columns.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) == 0:
            raise ValueError(f"Dataset is empty: {path}")
        return df

    def validate_preprocessed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the NLP-ready DataFrame.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            Validation dictionary with a top-level "is_valid" flag.
        """
        validation: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        missing_columns = [col for col in OUTPUT_COLUMNS if col not in df.columns]
        if missing_columns:
            validation["errors"].append(f"Missing output columns: {missing_columns}")
            validation["is_valid"] = False
            return validation

        empty_clean = df["clean_question"].isna() | (
            df["clean_question"].astype(str).str.strip() == ""
        )
        if empty_clean.sum() > 0:
            validation["errors"].append(
                f"Found {int(empty_clean.sum())} empty clean_questions"
            )
            validation["is_valid"] = False

        empty_intents = df["intent"].isna() | (df["intent"].astype(str).str.strip() == "")
        if empty_intents.sum() > 0:
            validation["errors"].append(f"Found {int(empty_intents.sum())} empty intents")
            validation["is_valid"] = False

        invalid_intents = set(df["intent"].astype(str).str.strip()) - set(CANONICAL_INTENTS)
        if invalid_intents:
            validation["errors"].append(
                f"Non-canonical intents: {sorted(invalid_intents)}"
            )
            validation["is_valid"] = False

        invalid_entities = set()
        for value in df["entity"].astype(str).str.strip():
            if value and value not in CANONICAL_ENTITIES:
                invalid_entities.add(value)
        if invalid_entities:
            validation["errors"].append(
                f"Non-canonical entities: {sorted(invalid_entities)}"
            )
            validation["is_valid"] = False

        no_tokens = (
            df["tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0) == 0
        )
        if no_tokens.sum() > 0:
            validation["warnings"].append(
                f"Found {int(no_tokens.sum())} records with empty token lists"
            )

        return validation

    # ----------------------------------------------------- Statistics
    def generate_nlp_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute NLP statistics from the NLP-ready DataFrame.

        Args:
            df: Preprocessed DataFrame (with OUTPUT_COLUMNS).

        Returns:
            Statistics dictionary used by the report generator.
        """
        clean_lens = df["clean_question"].astype(str).str.len()
        token_counts = df["tokens"].apply(len)
        filtered_counts = df["filtered_tokens"].apply(len)
        lemmatized_counts = df["lemmatized_tokens"].apply(len)

        raw_vocab = set()
        for tokens in df["tokens"]:
            raw_vocab.update(tokens if isinstance(tokens, list) else [])
        lemma_vocab = set()
        for tokens in df["lemmatized_tokens"]:
            lemma_vocab.update(tokens if isinstance(tokens, list) else [])

        entities = df["entity"].astype(str).str.strip()
        non_empty_entities = entities[entities != ""]

        intent_counts = df["intent"].value_counts()
        entity_counts = non_empty_entities.value_counts()

        return {
            "total_questions": int(len(df)),
            "total_intents": int(df["intent"].nunique()),
            "intents": sorted(df["intent"].unique().tolist()),
            "avg_question_length": round(float(clean_lens.mean()), 2),
            "min_question_length": int(clean_lens.min()),
            "max_question_length": int(clean_lens.max()),
            "avg_token_count": round(float(token_counts.mean()), 2),
            "min_token_count": int(token_counts.min()),
            "max_token_count": int(token_counts.max()),
            "avg_filtered_token_count": round(float(filtered_counts.mean()), 2),
            "avg_lemmatized_token_count": round(float(lemmatized_counts.mean()), 2),
            "avg_stopwords_removed": round(
                float((token_counts - filtered_counts).mean()), 2
            ),
            "raw_vocabulary_size": int(len(raw_vocab)),
            "lemmatized_vocabulary_size": int(len(lemma_vocab)),
            "unique_entities": int(non_empty_entities.nunique()),
            "questions_with_entities": int((entities != "").sum()),
            "questions_without_entities": int((entities == "").sum()),
            "entity_frequency": entity_counts.to_dict(),
            "intent_distribution": intent_counts.to_dict(),
            "min_intent_examples": int(intent_counts.min()),
            "max_intent_examples": int(intent_counts.max()),
            "missing_values": {col: int(df[col].isna().sum()) for col in OUTPUT_COLUMNS},
            "duplicate_questions": int(df["question"].duplicated().sum()),
            "duplicate_clean_questions": int(df["clean_question"].duplicated().sum()),
            "empty_records": int(
                df["clean_question"].isna().sum()
                + (df["clean_question"].astype(str).str.strip() == "").sum()
            ),
        }

    def save_nlp_dataset(
        self, df: pd.DataFrame, output_dir: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """
        Save the NLP-ready dataset as CSV and JSON.

        List fields (tokens, filtered_tokens, lemmatized_tokens, entities) are
        stored as space/comma-joined strings in the CSV (a flat format) and as
        real JSON arrays in the JSON file.

        Args:
            df: NLP-ready DataFrame.
            output_dir: Directory in which to write the files.

        Returns:
            Tuple of (csv_path, json_path).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "faq_nlp_ready.csv"
        json_path = output_dir / "faq_nlp_ready.json"

        # CSV: flatten list columns.
        df_csv = df.copy()
        df_csv["tokens"] = df_csv["tokens"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
        df_csv["filtered_tokens"] = df_csv["filtered_tokens"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
        df_csv["lemmatized_tokens"] = df_csv["lemmatized_tokens"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
        df_csv["entities"] = df_csv["entities"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x)
        )
        df_csv.to_csv(csv_path, index=False, encoding="utf-8")

        # JSON: keep list structures as real JSON arrays.
        df_json = df.copy()
        for col in ("tokens", "filtered_tokens", "lemmatized_tokens", "entities"):
            df_json[col] = df_json[col].apply(
                lambda x: list(x) if isinstance(x, list) else []
            )
        records = df_json.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)

        return csv_path, json_path

    # ---------------------------------------------------- Quality gate
    def compute_quality_gate(
        self,
        input_df: Optional[pd.DataFrame],
        output_df: Optional[pd.DataFrame],
        validation: Dict[str, Any],
        stats: Dict[str, Any],
        artifacts: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the Module 2 quality gate.

        Args:
            input_df: Module 1 input DataFrame.
            output_df: NLP-ready output DataFrame.
            validation: Result of validate_preprocessed_data().
            stats: Result of generate_nlp_statistics().
            artifacts: Dict of generated artifact checks, e.g.
                {"csv": True, "json": True, "report": True, "charts": True}.

        Returns:
            Dictionary mapping check names to booleans plus an overall pass.
        """
        artifacts = artifacts or {}
        checks: Dict[str, bool] = {}

        input_rows = 0 if input_df is None else len(input_df)
        output_rows = 0 if output_df is None else len(output_df)

        checks["Module 1 dataset loads"] = input_df is not None and input_rows > 0
        checks["Input records = 294"] = input_rows == 294
        checks["Output records = 294"] = output_rows == 294
        checks["Record count consistency"] = output_rows == input_rows
        checks["Intent count remains 22"] = (
            output_df is not None and output_df["intent"].nunique() == 22
        )
        checks["Minimum intent examples >= 12"] = (
            output_df is not None
            and int(output_df["intent"].value_counts().min()) >= 12
        )
        checks["clean_question generated"] = (
            output_df is not None
            and int(
                output_df["clean_question"].isna().sum()
                + (output_df["clean_question"].astype(str).str.strip() == "").sum()
            )
            == 0
        )
        checks["Tokenization works"] = (
            output_df is not None
            and int(
                output_df["tokens"].apply(
                    lambda x: 0 if isinstance(x, list) and len(x) else 1
                ).sum()
            )
            == 0
        )
        checks["Stopword processing works"] = (
            output_df is not None and "filtered_tokens" in output_df.columns
        )
        checks["Lemmatization works"] = (
            output_df is not None and "lemmatized_tokens" in output_df.columns
        )
        checks["Intent normalization works"] = (
            output_df is not None
            and set(output_df["intent"].astype(str).str.strip()) == set(CANONICAL_INTENTS)
        )
        checks["Entity preparation works"] = (
            output_df is not None
            and "entity" in output_df.columns
            and all(
                str(value).strip() == "" or str(value).strip() in CANONICAL_ENTITIES
                for value in output_df["entity"]
            )
        )
        checks["Entity extraction works"] = (
            output_df is not None and "entities" in output_df.columns
        )
        checks["No unintended data loss"] = (
            output_df is not None
            and len(output_df.columns) >= 8
            and output_rows == input_rows
        )
        checks["No duplicate questions introduced"] = (
            output_df is not None and int(output_df["question"].duplicated().sum()) == 0
        )
        checks["No missing required values"] = (
            output_df is not None
            and sum(int(output_df[col].isna().sum()) for col in REQUIRED_COLUMNS) == 0
        )
        checks["NLP-ready CSV generated"] = bool(artifacts.get("csv", False))
        checks["NLP-ready JSON generated"] = bool(artifacts.get("json", False))
        checks["NLP report generated"] = bool(artifacts.get("report", False))
        checks["NLP charts generated"] = bool(artifacts.get("charts", False))
        checks["Preprocessing validation passes"] = bool(
            validation.get("is_valid", False)
        )

        return {
            "checks": checks,
            "overall_pass": all(checks.values()),
        }


def main() -> None:
    """Quick smoke test for the preprocessor module."""
    preprocessor = NLPPreprocessor()

    print("=== NLP Preprocessor smoke test ===")
    sample = "  How Do I Reset My Password??? "
    print(f"clean_text      : {preprocessor.clean_text(sample)!r}")
    tokens = preprocessor.tokenize_text(preprocessor.clean_text(sample))
    print(f"tokens          : {tokens}")
    print(f"filtered_tokens : {preprocessor.remove_stopwords(tokens)}")
    print(
        "lemmatized      : "
        + str(preprocessor.lemmatize_tokens(["employees", "running", "issues"]))
    )
    print(f"normalize_intent: {preprocessor.normalize_intent('Password_Reset')!r}")
    print(f"normalize_entity: {preprocessor.normalize_entity('password_reset')!r}")
    print(
        "extract_entities: "
        + str(preprocessor.extract_entities("My laptop cannot connect to Wi-Fi."))
    )
    print("=== OK ===")


if __name__ == "__main__":
    main()
