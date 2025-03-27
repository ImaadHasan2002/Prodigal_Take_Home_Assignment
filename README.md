# The Customer-Agent Call Analyzer

**The Customer-Agent Call Analyzer** is a robust application designed to analyze conversations between debt collection agents and borrowers. It evaluates compliance, professionalism, and call quality metrics by processing conversation data stored in YAML (or JSON) files with detailed utterance-level information.

---

## Key Features

- **Profanity Detection:**  
  Identify calls where agents or borrowers used profane language using regex pattern matching or AI-based detection.

- **Privacy/Compliance Violation Detection:**  
  Detect instances where agents shared sensitive information without proper identity verification.

- **Call Quality Metrics:**  
  Calculate important metrics such as:
  - **Overtalk:** Percentage of the call where both parties speak simultaneously.
  - **Silence:** Percentage of the call where no one is speaking.
  - **Timeline Visualization:** Graphically displays speaking patterns throughout the call.

---

## Installation

### Prerequisites

- **Python 3.7+**
- **Streamlit** (for the web application)
- **PyYAML** (for YAML file parsing)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **Matplotlib** (for data visualization)
- **Seaborn** (for enhanced data visualization)
- **Scikit-learn** (for machine learning models)
- **Transformers** (for LLMs and advanced NLP tasks)
- **NLTK** (for natural language processing tasks)
- **Spacy** (for advanced NLP tasks)
- **TextBlob** (for sentiment analysis and text processing)
- **Regex** (for pattern matching)
- **pip** (Python package installer)

### Setup Instructions

1. **Clone the repository or download the project files:**

   ```bash
   git clone <repository-url>
   cd Prodigal_Take_Home_Assignment
    ```
2. **Create and install the required Python packages:**

   On Windows:

   ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    On macOS/Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the application by running the following command:
```bash
streamlit run app.py
```
The application will open in your default web browser.

## USAGE OF THE APPLICATION

## Usage Instructions

1. **Upload a Call File:**
   - Click **"Browse files"** in the sidebar.
   - Select a YAML or JSON file from your device.
   - Ensure the file follows the required structure with the fields: `speaker`, `text`, `stime`, and `etime`.

2. **Select Analysis Type:**
   - Choose one of the following options:
     - **Profanity Detection**
     - **Privacy & Compliance**
     - **Call Quality Metrics**

3. **Select Analysis Approach (for Profanity and Privacy/Compliance):**
   - **Pattern Matching (Regex):**  
     Uses regular expression patterns for quick, deterministic analysis.
   - **Machine Learning/LLM:**  
    Uses a pretrained model for more nuanced, context-aware analysis, leveraging advanced natural language processing techniques to better understand the subtleties of conversations.

4. **View Results:**
   - The analysis results will be displayed in the main panel.
   - For **Call Quality Metrics**, visualizations will show overtalk and silence percentages.


## Implementation Details

### Analysis Approaches

- **Pattern Matching (Regex):**
  - Uses predefined patterns to detect profanity, sensitive information, and verification attempts
  - Fast and deterministic but may miss nuanced context.
  - For profanity detection, matches against a predefined list of profane words.

- **LLM Prompting:**
  - Uses a Large Language Model with prompting strategies for profanity detection
  - Better at understanding context and nuanced language.
  - Falls back to regex detection if the LLM encounters an error.

### Call Quality Metrics

- **Overtalk:**  
  Percentage of the call where the agent and borrower speak simultaneously.

- **Silence:**  
  Percentage of the call where no one is speaking.

- **Timeline Visualization:**  
  Shows the speaking patterns throughout the call.

### Project Structure

Project Structure
prodigal_take_home_assignment/
├── app.py                  - Main Streamlit application
├── modules/                - Analysis modules
│   ├── profanity.py        - Profanity detection
│   ├── privacy.py          - Privacy/compliance violation detection
│   └── metrics.py          - Call quality metrics
├── utils/                  - Utility functions
│   └── loader.py           - File loading and parsing
├── data/                   - Data storage
│   ├── sample_calls/       - Sample call data
│   └── test_calls/         - Test calls (synthetically generated using LLMs)
├── requirements.txt        - Python dependencies
└── README.md               - Project documentation


## Sample Call Data

The application expects call data in YAML format with the following structure:

```yaml
utterances:
  - speaker: "Agent"
    text: "Hello, this is Mark from XYZ Collections. Am I speaking with Jessica?"
    stime: 0
    etime: 7

  - speaker: "Borrower"
    text: "Yes, this is Jessica. What is this about?"
    stime: 8
    etime: 13
    
  # Additional utterances...

  ```

  ## Recommendations

- **Profanity Detection:**  
  - The regex approach works well for detecting explicit terms.  
  - The LLM Prompting provides better context awareness for subtle cases but may be slower.

- **Privacy Compliance:**  
  - The Pretrained model approach is generally superior as it can better understand verification contexts.

- **Call Quality Metrics:**  
  - Use the timeline visualization to identify patterns in agent-borrower interactions.

---

## Troubleshooting

- **Dependencies:**  
  Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
    ```

- **File Format:**
    - Ensure the uploaded file is in YAML or JSON format.
    - Check the file structure and field names.

- **Analysis Errors:**
    - If the analysis fails, try a different approach (Regex vs. LLM Prompting).
    - Check the error messages for more information.

- **Application Issues:**
    - If the application crashes, restart the server with `streamlit run app.py`.