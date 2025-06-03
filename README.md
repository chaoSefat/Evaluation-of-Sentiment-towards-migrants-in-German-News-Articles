# Diachronic Sentiment Analysis of Migration Discourse (2010–2024)

**Author**: Abdullah Al Sefat  
**Status**: Independent research project  
**Topic**: Natural Language Processing, Computational Social Science, Bias Analysis

---

## 🧠 Overview

This project investigates how sentiment toward *refugees*, *asylum seekers*, and *immigrants* has changed over time in **English** and **German** news media, using sentence-level analysis powered by **GPT-4-mini**.

Using the **Leipzig Corpora Collection** (2010–2024), 26 Million ~+ sentences were filtered to obtain a subset of migration-related sentences and analyzed their sentiment to track evolving societal attitudes across two languages and over 15 years.

---

## 📚 Dataset

- **Corpus**: Leipzig Wortscathz Corpora Collection 
- **Languages**: English (`eng_news_20XX`) and German (`deu_news_20XX`)
- **Time Period**: 2010–2024 (excluding missing years: [2011, 2012, 2021, 2022] for English)
- **Data Format**: One txt file per year containing: *Number* >TAB< *Sentence*

## 🔍 Dataset Filtering Methodology

The raw Leipzig corpora required filtering to isolate migration-relevant content. A two-stage hybrid approach was implemented combining lexical pre-filtering with semantic similarity scoring:

### Semantic Similarity Framework
- **Model**: `sentence-transformers/gtr-t5-large` for high-dimensional sentence embeddings
- **Seed Sentences**: 9 German exemplars covering key migration concepts (*Flüchtlinge*, *Asylbewerber*, *Migration*, etc.)
- **Similarity Metric**: Cosine similarity with threshold ≥ 0.75
- **Processing**: Batch inference (128 sentences) with tensor normalization

### Two-Stage Filtering Pipeline
1. **Lexical Pre-filter**: Regex pattern matching on stemmed German keywords (`migr*`, `flücht*`, `asyl*`, `einwander*`, etc.)
2. **Semantic Validation**: Sentence embeddings compared against seed sentence representations using cosine similarity

This approach ensures both computational efficiency (keyword pre-filtering reduces semantic processing load) and precision (embeddings capture contextual nuances beyond keyword matching). The same methodology was applied to English corpora with translated seed sentences and corresponding English keyword patterns.



