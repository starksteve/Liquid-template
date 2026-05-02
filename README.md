# ⚡ DataForge AI — AI-Powered Data Engineering Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Free)-orange)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?logo=apachespark)
![License](https://img.shields.io/badge/license-MIT-green)

> End-to-end data engineering platform that turns plain English transformation descriptions into production-ready Scala/Spark, PySpark, and Spark SQL code — complete with data quality rules, unit tests, and data lineage diagrams.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🤖 AI Schema Suggest** | Upload source + target CSVs — AI automatically suggests transformation logic for each column |
| **⚡ Scala/Spark Code** | Production-ready Spark DataFrame transformation code with agentic self-correction |
| **🐍 PySpark Code** | Python/PySpark equivalent with type hints and `__main__` example |
| **🗄️ Spark SQL** | Delta Lake DDL, CREATE VIEW, and SELECT statements |
| **✅ Data Quality Rules** | Auto-generated Spark validation code (null checks, ranges, patterns) that runs before transformation |
| **🧪 Unit Tests** | ScalaTest (AnyFlatSpec) + pytest suites generated alongside the code |
| **🔗 Data Lineage** | Interactive SVG diagram: Source → Transform → Output |
| **📋 Pipeline README** | AI writes a full technical README.md for your pipeline |
| **📊 Data Profiler** | Deep CSV analysis — quality score, null %, cardinality, outliers, recommendations |
| **📦 Export ZIP** | All artifacts in one download: Scala, PySpark, SQL, tests, quality code, README |
| **📜 History** | Session history of all generation runs |

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────┐
│  📁 Source  │───▶│ ✅ Quality   │───▶│  ⚙️ Transform  │───▶│ 📤 Output│
│  CSV Upload │    │    Check     │    │  Code Gen (AI) │    │  Artifacts│
└─────────────┘    └──────────────┘    └────────────────┘    └──────────┘
                          │                     │
                    DQ Rules (Spark)     Scala + PySpark
                                         + SQL + Tests
                                         + Lineage + README
```

---

## ⚡ Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/your-username/dataforge-ai.git
cd dataforge-ai
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
```bash
# Copy the template
cp env.example .env

# Edit .env and add your key
# Get a FREE key at: https://console.groq.com
GROQ_API_KEY=your_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub (the `.gitignore` already excludes all secrets)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo → `app.py`
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
4. Click **Deploy** — your app is live 🎉

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Free API key from [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |

---

## 📁 Project Structure

```
dataforge-ai/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── env.example                     # Environment variables template
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   ├── config.toml                 # Streamlit theme & server config
│   └── secrets.toml.example        # Streamlit Cloud secrets template
├── utils/
│   ├── llm_config.py               # Groq LLM client setup
│   ├── scala_code_generator.py     # Scala/Spark code generation (agentic)
│   ├── pyspark_generator.py        # PySpark code generation
│   ├── sql_generator.py            # Spark SQL generation
│   ├── test_generator.py           # ScalaTest + pytest generation
│   ├── data_profiler.py            # CSV data quality profiling
│   ├── lineage_visualizer.py       # SVG data lineage diagram
│   ├── schema_suggester.py         # AI schema mapping suggester
│   ├── quality_rules_generator.py  # Spark DQ validation code generator
│   └── readme_generator.py         # AI pipeline README generator
├── data/
│   ├── GenderXwalk.csv             # Sample crosswalk table
│   └── LanguageXwalk.csv           # Sample crosswalk table
└── transformation/                 # Sample transformation files
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| Streamlit | Web UI framework |
| Groq API (LLaMA 3.3 70B) | AI/LLM code generation — **free** |
| LangChain | LLM abstraction layer |
| pandas | Data profiling & CSV handling |
| Apache Spark (target) | Runtime for generated code |

---

## 📄 License

MIT — free to use, modify, and distribute.


## Features

- **Input Support**: CSV and TXT files or raw text input
- **Output Schema**: Optional JSON schema or example output format
- **AI-Powered Generation**: Uses Azure OpenAI to generate Liquid templates
- **Agentic Workflow**: Automatically refines templates based on rendering errors
- **User-Configurable Iterations**: Control how many times the AI attempts to fix template errors (1-10 iterations)
- **Progress Tracking**: Real-time progress updates during template generation
- **Interactive UI**: Clean Streamlit interface with real-time feedback
- **Template Editing**: User can edit generated templates and re-render
- **Debug Output**: Comprehensive logging for troubleshooting
- **Sidebar Configuration**: Easy access to settings and advanced options

## Setup Instructions

1. **Install Dependencies**
   ```bash
   # Activate your virtual environment
   source liquidgenerator_env/bin/activate  # On macOS/Linux
   # or
   liquidgenerator_env\Scripts\activate     # On Windows
   
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   
   Set up your Azure OpenAI credentials. You can either:
   
   **Option A: Set environment variables directly**
   ```bash
   export AZURE_OPENAI_API_KEY="your_api_key_here"
   export ENDPOINT_URL="your_endpoint_url_here"
   export DEPLOYMENT_NAME="your_deployment_name_here"
   export API_VERSION="2025-01-01-preview"
   ```
   
   **Option B: Create a .env file**
   ```bash
   # Create .env file in the project root
   touch .env
   
   # Add your credentials to .env file:
   AZURE_OPENAI_API_KEY=your_api_key_here
   ENDPOINT_URL=your_endpoint_url_here
   DEPLOYMENT_NAME=your_deployment_name_here
   API_VERSION=2025-01-01-preview
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Usage

### Configuration (Sidebar)
- **Maximum AI Iterations**: Use the slider to set how many times (1-10) the AI should attempt to fix template errors
- **Debug Information**: Toggle to show/hide detailed debug information in the app interface

### Main Interface
1. **Input Data**: 
   - Paste CSV or text data in the left text area, OR
   - Upload a CSV or TXT file

2. **Output Format** (Optional):
   - Paste a JSON schema or example output in the right text area, OR
   - Upload a JSON file with the desired format

3. **Additional Instructions** (Optional):
   - Provide specific formatting or transformation instructions

4. **Generate Template**:
   - Click "Generate Liquid Template" to start the AI generation process
   - Watch the progress bar and status updates as the AI works
   - The system will automatically refine the template if rendering fails

5. **Edit and Re-render**:
   - Edit the generated template if needed
   - Click "Render Template" to see the updated output

### Progress Tracking
- **Progress Bar**: Shows the current iteration progress
- **Status Updates**: Real-time status messages during generation
- **Iteration Counter**: Displays current iteration out of maximum
- **Statistics**: View template length, iterations used, and output length

## Project Structure

```
LiquidTemplateGenerator/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── liquidgenerator_env/   # Virtual environment
├── templates/             # Template storage (optional)
└── utils/                 # Utility modules
    ├── __init__.py
    ├── llm_config.py      # Azure OpenAI configuration
    ├── template_generator.py  # AI template generation logic
    ├── file_handler.py    # File processing utilities
    └── template_renderer.py   # Liquid template rendering
```

## How It Works

1. **Input Processing**: The application accepts CSV or text data through text input or file upload
2. **Template Generation**: Azure OpenAI generates an initial Liquid template based on the input data and optional output schema
3. **Validation Loop**: The system attempts to render the template with the input data
4. **Refinement**: If rendering fails, the error details are sent back to the AI for template refinement
5. **Progress Updates**: Real-time progress tracking shows current iteration and status
6. **Iteration Control**: Users can set the maximum number of refinement attempts (1-10)
7. **User Editing**: Users can manually edit the final template and re-render as needed

## Configuration Options

### Iteration Settings
- **Range**: 1-10 iterations
- **Default**: 5 iterations
- **Purpose**: Controls how many times the AI will attempt to fix template errors before stopping
- **Recommendation**: 
  - Use 1-3 for simple data transformations
  - Use 4-6 for moderate complexity (default)
  - Use 7-10 for complex transformations or strict output requirements

### Debug Settings
- **Show Debug Information**: Toggle detailed debug output in the app interface
- **Console Logging**: Always active for terminal/console debugging

## Dependencies

- streamlit>=1.28.0
- langchain-openai>=0.1.0
- openai>=1.0.0
- pandas>=2.0.0
- python-dotenv>=1.0.0
- python-liquid>=1.9.0
- jsonschema>=4.17.0
- pyyaml>=6.0
- requests>=2.31.0

## Troubleshooting

- **Debug Output**: Check the console/terminal for detailed DEBUG messages or enable "Show debug information" in the sidebar
- **API Errors**: Verify your Azure OpenAI credentials and endpoint
- **Template Errors**: Increase max iterations if templates are failing to generate correctly
- **Performance**: Reduce max iterations if generation is taking too long
- **File Upload Issues**: Ensure files are in supported formats (CSV, TXT, JSON)

## Example Usage

1. **CSV Input**:
   ```csv
   name,age,city
   John,25,New York
   Jane,30,Los Angeles
   ```

2. **JSON Output Schema**:
   ```json
   {
     "people": [
       {"fullName": "string", "years": "number", "location": "string"}
     ]
   }
   ```

3. **Configuration**:
   - Set max iterations to 6 for reliable generation
   - Enable debug information if you want to see detailed progress

4. **Generated Template** (example):
   ```liquid
   {
     "people": [
       {% for row in rows %}
       {
         "fullName": "{{ row.name }}",
         "years": {{ row.age }},
         "location": "{{ row.city }}"
       }{% unless forloop.last %},{% endunless %}
       {% endfor %}
     ]
   }
   ```

## Tips for Best Results

1. **Start with fewer iterations** (2-3) for simple transformations
2. **Use more iterations** (6-8) when working with complex data structures
3. **Enable debug mode** when troubleshooting generation issues
4. **Provide clear output schemas** for better template accuracy
5. **Use specific additional instructions** for custom formatting requirements