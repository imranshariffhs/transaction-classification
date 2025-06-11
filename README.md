# Transaction Classification System

## Overview

This system classifies financial transactions by associating them with the correct organization and source using a multi-layered, intelligent approach. It combines local databases with external knowledge to ensure accurate and up-to-date classification.

## How It Works

### Primary Lookup (PostgreSQL Master Table)  
Transactions are first matched against a master organization table stored in PostgreSQL. This table contains verified organizations and categories for quick classification.

### Semantic Search (PGVector)  
If no exact match is found, the system performs a semantic search using PGVector embeddings to find similar organizations based on contextual similarity.

### External Enrichment (Google Search API & Gemini via LangChain)  
When local data is insufficient, external APIs like Google Search and LLM tools such as Gemini (via LangChain) are queried. These tools provide real-world context to identify unknown organizations.

### Continuous Learning  
New organizations and classifications found via external searches are automatically added to the vector database. This enriches the system’s knowledge base and improves classification accuracy over time.

## Technologies

- **LangChain:** Orchestration and LLM integration  
- **Gemini:** Intelligent classification and enrichment  
- **PGVector:** Semantic vector similarity search  
- **PostgreSQL:** Master data storage  
- **Google Search API:** Real-world data validation and enrichment  

## Benefits

- Accurate classification even with unknown or ambiguous data  
- Continuous improvement through automatic knowledge updates  
- Integration of multiple data sources and AI models
