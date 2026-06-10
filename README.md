# Progetto Information Retrieval e Natural Language Processing

## Obiettivo
Analisi di modelli linguistici generativi (LLM) per la generazione di risposte a tematiche psicologiche, supportata da tecniche di clustering semantico, sentiment analysis, classificazione e fine-tuning.

## Fasi di Analisi

**1. Preprocessing, Clustering Semantico e Sentiment Analysis**
* `1_nlp-clustering-semantico-sentiment-analysis.ipynb`: Fase iniziale di pulizia e preprocessing del dataset di Q&A psicologiche. Applicazione di tecniche di raggruppamento (clustering) per aggregare semanticamente le domande in macro-categorie. Include l'analisi del sentiment per determinare la polarità emotiva dei testi.

**2. Classificazione dei Disturbi**
* `2_nlp-classificazione-risposte.ipynb`: Utilizzo di modelli generativi per estrarre e classificare il disturbo primario del paziente a partire dalle risposte testuali fornite dagli esperti. I risultati sono impiegati per arricchire il dataset con nuove variabili categoriali.

**3. Generazione delle Risposte Pre-Finetuning**
* `3_nlp-generazione-risposte-senza-finetuing.ipynb`: Valutazione zero-shot. Test delle capacità di base dei LLM nel rispondere ai quesiti psicologici senza alcun addestramento dominio-specifico. Stabilisce la baseline prestazionale.

**4. Fine-Tuning dei Modelli (QLoRA)**
Script dedicati all'adattamento efficiente dei parametri (Parameter-Efficient Fine-Tuning) di modelli *state-of-the-art* sfruttando il dataset psicologico:
* `gemma3-finetuning.ipynb`: Addestramento del modello Google Gemma 3.
* `llama-finetuning.ipynb`: Addestramento del modello Meta LLaMA.
* `qwen-finetuning.ipynb`: Addestramento del modello Qwen.

**5. Valutazione Post-Finetuning**
* `risposte_modelli_post_finetuning.ipynb`: Generazione finale e valutazione quantitativa. Misurazione dei miglioramenti sfruttando metriche avanzate di Natural Language Generation (es. BLEU, BERTScore, MAUVE) per validare la coerenza semantica e sintattica delle reti neurali addestrate.

**6. Documentazione Completa**
* `relazioneNLP.pdf`: Documento tecnico-teorico del progetto. Contiene la formalizzazione matematica, i dettagli architetturali, i grafici di valutazione e il confronto critico tra le prestazioni dei vari modelli analizzati.

## Tecnologie e Librerie Principali
* `transformers`, `peft`, `trl`, `bitsandbytes` (per l'addestramento e quantizzazione LLM).
* `sentence-transformers`, `evaluate`, `nltk`, `bert_score` (per embedding e valutazione metrica).
* `pandas`, `numpy`, `scikit-learn` (per la manipolazione dati e clustering ML).
