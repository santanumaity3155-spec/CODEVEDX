"""
NLP Preprocessing Module for Internal Helpdesk Chatbot
Module 2: NLP Text Preprocessing + Intent/Entity Preparation

This module provides reusable functions for:
- Text normalization and cleaning
- Tokenization
- Stopword handling
- Lemmatization
- Intent normalization
- Entity extraction and validation
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

# Try to import NLTK, handle gracefully if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class NLPPreprocessor:
    """Handles NLP preprocessing for FAQ dataset."""
    
    # Canonical entity list for internal helpdesk
    CANONICAL_ENTITIES = [
        "password", "account", "laptop", "wifi", "email",
        "leave", "attendance", "salary", "holiday", "payroll",
        "employee_id", "location", "software", "security", "hr",
        "it_support", "internet", "vpn", "mobile", "printer"
    ]
    
    def __init__(self, project_root=None):
        """Initialize with project root directory."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Initialize NLTK components if available
        self.lemmatizer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK resources."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            
            # Keep important question words that affect intent
            important_words = {
                'how', 'what', 'when', 'where', 'why', 'who', 
                'can', 'could', 'would', 'should', 'do', 'does',
                'did', 'will', 'shall', 'may', 'might', 'must'
            }
            self.stop_words = self.stop_words - important_words
            
        except Exception as e:
            print(f"Warning: NLTK initialization failed: {e}")
            NLTK_AVAILABLE = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Convert to string safely
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove control characters (except newline and tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize punctuation: remove excessive punctuation but keep single instances
        # Remove question marks at the end (common in questions)
        text = re.sub(r'\?+\s*$', '', text)
        # Remove excessive exclamation marks
        text = re.sub(r'!{2,}', '!', text)
        # Remove excessive periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove special characters but keep alphanumeric, spaces, and basic punctuation
        # Keep: letters, numbers, spaces, apostrophes, hyphens
        text = re.sub(r"[^a-z0-9\s'-]", '', text)
        
        # Normalize whitespace again after cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                return tokens
            except Exception:
                # Fallback to simple split
                return text.split()
        else:
            # Simple fallback tokenization
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens with stopwords removed
        """
        if not NLTK_AVAILABLE or not self.stop_words:
            # If NLTK not available, return tokens as-is
            return tokens
        
        filtered = [token for token in tokens if token.lower() not in self.stop_words]
        return filtered
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not NLTK_AVAILABLE or not self.lemmatizer:
            # If NLTK not available, return tokens as-is
            return tokens
        
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def normalize_intent(self, intent: str) -> str:
        """
        Normalize intent label to canonical form.
        
        Args:
            intent: Intent label to normalize
            
        Returns:
            Normalized intent label
        """
        if not isinstance(intent, str):
            return ""
        
        # Convert to string and strip
        intent = str(intent).strip()
        
        # Convert to lowercase
        intent = intent.lower()
        
        # Replace spaces with underscores
        intent = re.sub(r'\s+', '_', intent)
        
        # Remove any special characters except underscores
        intent = re.sub(r'[^a-z0-9_]', '', intent)
        
        # Remove consecutive underscores
        intent = re.sub(r'_+', '_', intent)
        
        # Remove leading/trailing underscores
        intent = intent.strip('_')
        
        return intent
    
    def extract_entities(self, text: str, existing_entity: Optional[str] = None) -> List[str]:
        """
        Extract entities from text using keyword matching.
        
        Args:
            text: Input text
            existing_entity: Existing entity from dataset (if any)
            
        Returns:
            List of detected entities
        """
        entities = []
        text_lower = text.lower()
        
        # If existing entity is valid, use it
        if existing_entity and existing_entity.strip():
            entity_clean = existing_entity.strip().lower()
            if entity_clean in self.CANONICAL_ENTITIES:
                entities.append(entity_clean)
        
        # Keyword-based entity extraction
        entity_keywords = {
            'password': ['password', 'passcode', 'pwd'],
            'account': ['account', 'login', 'username', 'profile'],
            'laptop': ['laptop', 'computer', 'pc', 'machine'],
            'wifi': ['wifi', 'wi-fi', 'wireless', 'network'],
            'email': ['email', 'mail', 'outlook', 'inbox'],
            'leave': ['leave', 'vacation', 'holiday', 'time off'],
            'attendance': ['attendance', 'timesheet', 'time sheet'],
            'salary': ['salary', 'pay', 'wage', 'compensation'],
            'payroll': ['payroll', 'paycheck', 'direct deposit'],
            'employee_id': ['employee id', 'emp id', 'staff id', 'id card'],
            'location': ['location', 'office', 'building', 'room'],
            'software': ['software', 'application', 'app', 'program'],
            'security': ['security', 'secure', 'breach', 'incident'],
            'hr': ['hr', 'human resources', 'personnel'],
            'it_support': ['it support', 'technical support', 'helpdesk', 'help desk'],
            'internet': ['internet', 'web', 'browsing'],
            'vpn': ['vpn', 'virtual private network'],
            'mobile': ['mobile', 'phone', 'smartphone', 'cell'],
            'printer': ['printer', 'printing', 'scanner']
        }
        
        for entity, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if entity not in entities:
                        entities.append(entity)
                    break
        
        return entities if entities else []
    
    def preprocess_question(self, question: str, intent: str, answer: str, 
                           entity: Optional[str] = None) -> Dict:
        """
        Preprocess a single question.
        
        Args:
            question: Original question
            intent: Intent label
            answer: Answer text
            entity: Optional entity
            
        Returns:
            Dictionary with preprocessed data
        """
        # Clean question
        clean_question = self.clean_text(question)
        
        # Tokenize
        tokens = self.tokenize_text(clean_question)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
        
        # Normalize intent
        normalized_intent = self.normalize_intent(intent)
        
        # Extract entities
        extracted_entities = self.extract_entities(clean_question, entity)
        
        return {
            'question': question,
            'clean_question': clean_question,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'intent': normalized_intent,
            'answer': answer,
            'entity': extracted_entities[0] if extracted_entities else (entity.strip() if entity and entity.strip() else ""),
            'entities': extracted_entities
        }
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire dataset.
        
        Args:
            df: Input DataFrame with question, intent, answer, entity columns
            
        Returns:
            Preprocessed DataFrame
        """
        if len(df) == 0:
            raise ValueError("Empty dataset provided")
        
        # Validate required columns
        required_columns = ['question', 'intent', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preprocess each row
        processed_data = []
        for idx, row in df.iterrows():
            entity = row.get('entity', '')
            if pd.isna(entity):
                entity = ''
            
            processed = self.preprocess_question(
                question=row['question'],
                intent=row['intent'],
                answer=row['answer'],
                entity=entity
            )
            processed_data.append(processed)
        
        # Create DataFrame
        df_processed = pd.DataFrame(processed_data)
        
        return df_processed
    
    def validate_preprocessed_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate preprocessed dataset.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = [
            'question', 'clean_question', 'tokens', 'filtered_tokens',
            'lemmatized_tokens', 'intent', 'answer', 'entity'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing columns: {missing_columns}")
        
        # Check for empty clean_question
        empty_questions = df['clean_question'].isna() | (df['clean_question'].astype(str).str.strip() == '')
        if empty_questions.sum() > 0:
            validation['errors'].append(f"Found {empty_questions.sum()} empty clean_questions")
            validation['is_valid'] = False
        
        # Check for empty intents
        empty_intents = df['intent'].isna() | (df['intent'].astype(str).str.strip() == '')
        if empty_intents.sum() > 0:
            validation['errors'].append(f"Found {empty_intents.sum()} empty intents")
            validation['is_valid'] = False
        
        # Check for valid intent labels (no special chars except underscore)
        invalid_intents = df[df['intent'].astype(str).str.contains(r'[^a-z0-9_]', regex=True, na=False)]
        if len(invalid_intents) > 0:
            validation['warnings'].append(f"Found {len(invalid_intents)} intents with special characters")
        
        return validation
    
    def generate_intent_metadata(self, df: pd.DataFrame) -> Dict:
        """
        Generate intent metadata from dataset.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Intent metadata dictionary
        """
        intents = {}
        
        for intent_name in df['intent'].unique():
            intent_data = df[df['intent'] == intent_name]
            
            # Get example questions
            examples = intent_data['clean_question'].head(5).tolist()
            
            # Get associated entities
            entities = intent_data['entity'].unique().tolist()
            entities = [e for e in entities if e and str(e).strip()]
            
            intents[intent_name] = {
                'intent': intent_name,
                'description': f"Questions about {intent_name.replace('_', ' ')}",
                'examples': examples,
                'count': len(intent_data),
                'entities': entities
            }
        
        return intents
    
    def generate_nlp_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate NLP preprocessing statistics.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_questions': len(df),
            'total_intents': df['intent'].nunique(),
            'intents': sorted(df['intent'].unique().tolist()),
            
            # Question length statistics
            'avg_question_length': df['clean_question'].str.len().mean(),
            'min_question_length': df['clean_question'].str.len().min(),
            'max_question_length': df['clean_question'].str.len().max(),
            
            # Token statistics
            'avg_token_count': df['tokens'].apply(len).mean(),
            'min_token_count': df['tokens'].apply(len).min(),
            'max_token_count': df['tokens'].apply(len).max(),
            'avg_filtered_token_count': df['filtered_tokens'].apply(len).mean(),
            'avg_lemmatized_token_count': df['lemmatized_tokens'].apply(len).mean(),
            
            # Vocabulary statistics
            'raw_vocabulary_size': len(set(word for tokens in df['tokens'] for word in tokens)),
            'lemmatized_vocabulary_size': len(set(word for tokens in df['lemmatized_tokens'] for word in tokens)),
            
            # Stopword statistics
            'avg_stopwords_removed': (df['tokens'].apply(len) - df['filtered_tokens'].apply(len)).mean(),
            
            # Entity statistics
            'questions_with_entities': (df['entity'].astype(str).str.strip() != '').sum(),
            'questions_without_entities': (df['entity'].astype(str).str.strip() == '').sum(),
            'entity_frequency': df['entity'].value_counts().to_dict(),
            
            # Intent distribution
            'intent_distribution': df['intent'].value_counts().to_dict(),
            
            # Data quality
            'empty_records': df['clean_question'].isna().sum() + (df['clean_question'].astype(str).str.strip() == '').sum(),
            'duplicate_clean_questions': df.duplicated(subset=['clean_question'], keep=False).sum()
        }
        
        return stats
    
    def save_nlp_dataset(self, df: pd.DataFrame, output_dir: Path):
        """
        Save NLP-ready dataset in multiple formats.
        
        Args:
            df: Preprocessed DataFrame
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV with list columns as strings
        df_csv = df.copy()
        df_csv['tokens'] = df_csv['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        df_csv['filtered_tokens'] = df_csv['filtered_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        df_csv['lemmatized_tokens'] = df_csv['lemmatized_tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        df_csv['entities'] = df_csv.get('entities', '').apply(lambda x: '|'.join(x) if isinstance(x, list) and x else '')
        
        csv_path = output_dir / "faq_nlp_ready.csv"
        df_csv.to_csv(csv_path, index=False)
        print(f"✓ Saved NLP-ready CSV: {csv_path}")
        
        # Save JSON (preserves list structures)
        json_path = output_dir / "faq_nlp_ready.json"
        df_json = df.copy()
        df_json['tokens'] = df_json['tokens'].apply(lambda x: x if isinstance(x, list) else [])
        df_json['filtered_tokens'] = df_json['filtered_tokens'].apply(lambda x: x if isinstance(x, list) else [])
        df_json['lemmatized_tokens'] = df_json['lemmatized_tokens'].apply(lambda x: x if isinstance(x, list) else [])
        df_json['entities'] = df_json.get('entities', '').apply(lambda x: x if isinstance(x, list) else [])
        
        records = df_json.to_dict(orient='records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved NLP-ready JSON: {json_path}")
        
        return csv_path, json_path


def main():
    """Main entry point for testing."""
    preprocessor = NLPPreprocessor()
    
    # Test with sample text
    sample = "  How Do I Reset My Password???  "
    print(f"Original: '{sample}'")
    print(f"Cleaned: '{preprocessor.clean_text(sample)}'")
    
    cleaned = preprocessor.clean_text(sample)
    print(f"Tokens: {preprocessor.tokenize_text(cleaned)}")
    print(f"Filtered: {preprocessor.remove_stopwords(preprocessor.tokenize_text(cleaned))}")
    print(f"Lemmatized: {preprocessor.lemmatize_tokens(preprocessor.remove_stopwords(preprocessor.tokenize_text(cleaned)))}")
    
    # Test intent normalization
    print(f"\nIntent normalization:")
    print(f"  'Password_Reset' -> '{preprocessor.normalize_intent('Password_Reset')}'")
    print(f"  'password_reset ' -> '{preprocessor.normalize_intent('password_reset ')}'")
    print(f"  'PASSWORD_RESET' -> '{preprocessor.normalize_intent('PASSWORD_RESET')}'")
    
    # Test entity extraction
    print(f"\nEntity extraction:")
    print(f"  'How do I reset my password?' -> {preprocessor.extract_entities('How do I reset my password?')}")
    print(f"  'My laptop cannot connect to Wi-Fi' -> {preprocessor.extract_entities('My laptop cannot connect to Wi-Fi')}")


if __name__ == "__main__":
    main()