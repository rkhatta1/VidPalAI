Of course. Here is a comprehensive summary of the project's goals, architecture, current status, and future plans, designed to provide context to an AI coding agent.

***
## ## Project Brief: AI Multi-Camera Podcast Editing Agent

### **High-Level Goal**
The project's mission is to create an AI-powered post-production assistant that analyzes multi-camera podcast footage (audio and video) and generates a professional Edit Decision List (EDL). This EDL can be imported into video editing software like Adobe Premiere Pro to create a fully-cut timeline, dramatically reducing the manual labor of scrubbing through and selecting camera angles.

### **Core Technologies**
* **VLM (Visual Analysis):** `apple/FastVLM-0.5B` (quantized to 8-bit) for generating timestamped descriptions of the visual feed.
* **Audio Transcription:** `openai-whisper` for generating a high-precision transcript with word-level timestamps.
* **LLM (Reasoning):** Local open-source models (e.g., `gemma2:2b`, `phi3`) run via the `Ollama` service.
* **Memory/RAG Framework:** `LlamaIndex` is used to create and query a persistent, searchable memory of the podcast.
* **Embedding Model:** `BAAI/bge-small-en-v1.5` for converting text chunks into vector embeddings for semantic search.
* **Vector Store:** `ChromaDB` for persistent, on-disk storage of the vector index.
* **Primary Libraries:** `transformers`, `torch`, `opencv-python`, `ollama`, `llama-index`, `chromadb`, `sentence-transformers`.

---
### ## Project Architecture: The Multi-Pass Workflow
The entire editing process is broken down into three distinct, sequential passes to mimic a human production workflow.

* **Pass 1: The "Producer" (High-Level Analysis)**
    * **Goal:** Create a structural map of the entire podcast.
    * **Process:** The full transcript is fed to an LLM, which segments the conversation into logical chapters based on topics and narrative shifts.
    * **Output:** A JSON file (`structural_map.json`) containing a list of chapters, each with a title, summary, and start/end timestamps.

* **Pass 2: The "Director" (Segment-by-Segment Editing)**
    * **Goal:** Make detailed, second-by-second camera cut decisions for each chapter.
    * **Process:** The agent iterates through the chapters from Pass 1. For each chapter, it prompts an LLM with a combination of **global context** (retrieved from the RAG index) and **local context** (the granular VLM descriptions and transcript for that specific chapter).
    * **Output:** A JSON file (`director_edits.json`) containing a detailed Edit Decision List for each chapter.

* **Pass 3: The "Finishing Editor" (Stitching & Polishing)**
    * **Goal:** Combine the chapter edits into a single timeline and smooth the transitions.
    * **Process:** An agent programmatically stitches the EDLs from Pass 2 together. It can optionally use an LLM to analyze the cut points between chapters and make final adjustments.
    * **Output:** A final, complete EDL in a standard format like **Final Cut Pro XML (FCPXML)**.

---
### ## Current Status & Completed Work

**Phase 1 (Data Ingestion and Analysis) is complete and functional:**

1.  **Audio/Video Pre-processing (`main.py`, `audio_processor.py`, `video_processor.py`):**
    * The pipeline can successfully process the first 10 minutes of a video (`.mp4`) and audio (`.mp3`) file.
    * `video_processor.py` uses a stable, single-threaded approach to extract frames at 1-second intervals and generate visual descriptions using the 8-bit quantized `FastVLM` model.
    * `audio_processor.py` uses `Whisper` to generate a word-level transcript.
    * The combined, timestamped granular data is saved to `output/processed_data_10min.json`.

2.  **RAG Index Population (`rag_agent.py`):**
    * A script successfully loads the `processed_data_10min.json`.
    * It chunks the data into 10-second "moments," combining visual and transcript information into LlamaIndex `Document` objects.
    * It builds a persistent `VectorStoreIndex` backed by `ChromaDB`, saving it to the `./llama_index_storage` directory. This process works reliably.

3.  **Producer Pass Implementation (`producer_pass.py`):**
    * This script, implementing Pass 1, is complete.
    * It successfully reads the full transcript, prompts the `gemma2:2b` model via Ollama, and generates the `output/structural_map.json` file with a correct chapter structure.

### ## Next Steps: Implement the Director & Finishing Passes

The immediate next step is to **implement Pass 2**.

* **Create `director_pass.py`:** This script will:
    1.  Load the chapters from `structural_map.json`.
    2.  Initialize the LlamaIndex RAG agent from the `./llama_index_storage` directory.
    3.  Loop through each chapter. For each one:
        * Perform a RAG query using the chapter's summary to get relevant global context.
        * Gather the granular local data from `processed_data_10min.json`.
        * Construct a detailed prompt for the "Director" LLM.
        * Call the Ollama LLM (`gemma2:2b`) with `format='json'` to get the EDL for that chapter.
    4.  Aggregate the EDLs and save them to `output/director_edits.json`.

After Pass 2 is complete, the final steps will be to implement **Pass 3** and the **XML converter**.
