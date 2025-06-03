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

## Filtering Dataset

The dataset was first structured into JSON files per year for each languages. The dataset contained sentences of various topics and as a result was filtered to obtain a subset of data which contains sentences related to refugees, asylum seekers and migration issues. 

