"""
TruthLens - NLP Preprocessor and spaCy Named Entity Recognition (NER) Pipeline
Handles text normalization, cleaning, lemmatization, entity extraction, and claim keyword generation.
"""

import re
import spacy
from typing import Dict, List, Any

# Load spaCy English core pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Cleans raw text:
    - Removes URLs and hyperlinks
    - Removes HTML tags
    - Replaces consecutive whitespace with a single space
    - Strips leading and trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize fancy unicode quotes & dashes
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('—', '-').replace('–', '-')
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Normalizes and lemmatizes text using spaCy.
    Reduces words to their base lemma and removes punctuation/stopwords if requested.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    
    lemmas = []
    for token in doc:
        # Ignore whitespace and punctuation
        if token.is_space or token.is_punct:
            continue
        if remove_stopwords and token.is_stop:
            continue
        # Preserve lemma in lowercase
        lemma = token.lemma_.lower().strip()
        if lemma:
            lemmas.append(lemma)
            
    return " ".join(lemmas)


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extracts Named Entities using spaCy (PERSON, ORG, GPE, DATE, EVENT, etc.).
    Returns categorized lists and raw entity objects.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    
    entity_dict = {
        "PERSON": set(),
        "ORG": set(),
        "GPE": set(),  # Geopolitical entity (countries, cities, states)
        "DATE": set(),
        "EVENT": set(),
        "OTHER": set()
    }
    
    all_entities = []
    
    for ent in doc.ents:
        label = ent.label_
        ent_text = ent.text.strip()
        
        all_entities.append({
            "text": ent_text,
            "label": label,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
        if label in entity_dict:
            entity_dict[label].add(ent_text)
        elif label in ["LOC", "FAC"]:
            entity_dict["GPE"].add(ent_text)
        elif label in ["NORP", "PRODUCT", "WORK_OF_ART"]:
            entity_dict["OTHER"].add(ent_text)
            
    # Convert sets to sorted lists for JSON serialization
    summary = {k: sorted(list(v)) for k, v in entity_dict.items() if len(v) > 0}
    
    return {
        "entities_by_type": summary,
        "all_entities": all_entities,
        "people": sorted(list(entity_dict["PERSON"])),
        "organizations": sorted(list(entity_dict["ORG"])),
        "locations": sorted(list(entity_dict["GPE"])),
        "dates": sorted(list(entity_dict["DATE"])),
        "total_entities_found": len(all_entities)
    }


def extract_claims_and_keywords(text: str, top_k_keywords: int = 6) -> Dict[str, Any]:
    """
    Extracts core claims, noun chunks, and search keywords for RAG query formation.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    
    # 1. Extract informative noun phrases
    noun_chunks = []
    for chunk in doc.noun_chunks:
        chunk_clean = chunk.text.strip()
        # Filter out short or stopword-only chunks
        if len(chunk_clean.split()) <= 4 and not chunk.root.is_stop:
            noun_chunks.append(chunk_clean)
            
    # Deduplicate while preserving order
    unique_noun_chunks = list(dict.fromkeys(noun_chunks))
    
    # 2. Extract salient keywords (ranked by POS and length)
    keywords = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.pos_ in ["PROPN", "NOUN", "VERB", "ADJ"] and len(token.text) > 2:
            keywords.append(token.lemma_.lower())
            
    # Count frequency and select top-k
    freq = {}
    for kw in keywords:
        freq[kw] = freq.get(kw, 0) + 1
    sorted_keywords = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:top_k_keywords]
    
    # 3. Primary Claim Candidate (First sentence or root clause)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    main_claim = sentences[0] if sentences else cleaned[:150]
    
    return {
        "main_claim": main_claim,
        "key_phrases": unique_noun_chunks[:5],
        "keywords": sorted_keywords,
        "search_query": " ".join(sorted_keywords[:4]) if sorted_keywords else main_claim
    }


def full_nlp_pipeline(text: str) -> Dict[str, Any]:
    """
    Runs the complete preprocessing and linguistic analysis suite.
    """
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    ner_data = extract_entities(cleaned)
    claim_data = extract_claims_and_keywords(cleaned)
    
    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "lemmatized_text": lemmatized,
        "ner": ner_data,
        "claim_analysis": claim_data
    }


if __name__ == "__main__":
    sample = "Jennifer Aniston and Brad Pitt secretly remarried in Paris according to InTouch Weekly."
    print("Testing Preprocessor Pipeline...")
    res = full_nlp_pipeline(sample)
    print("Cleaned:", res["cleaned_text"])
    print("Lemmatized:", res["lemmatized_text"])
    print("Entities:", res["ner"]["entities_by_type"])
    print("Claim Keywords:", res["claim_analysis"]["keywords"])
