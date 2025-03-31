# AI Tender Generator

An intelligent web application that automatically generates professional tender documents using AI, powered by OpenAI's GPT-4 and Pinecone vector database.

## Overview

The AI Tender Generator is a Streamlit-based application that helps organizations create comprehensive tender documents efficiently. It leverages machine learning to generate professional, compliant tender documents by learning from a database of existing tenders.

## Key Features

- **AI-Powered Generation:** Utilizes OpenAI's GPT-4 for generating contextually relevant tender content.
- **Similarity Search:** Uses Pinecone vector database to find and learn from similar existing tenders.
- **Structured Output:** Generates complete tender documents with standard sections.
- **User-Friendly Interface:** Simple web interface for inputting project details.
- **Multiple Export Options:** Download generated tenders in both TEXT and JSON formats.
- **Real-time Progress Tracking:** Visual feedback during document generation.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Pinecone API key
- Required Python packages (see `requirements.txt`)

## Installation

### Clone the repository:

```sh
git clone https://github.com/yourusername/ai-tender-generator.git
cd ai-tender-generator
```

### Install required packages:

```sh
pip install -r requirements.txt
```

### Set up environment variables:

Create a `.streamlit/secrets.toml` file with:

```toml
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
```

## Usage

### Start the Streamlit application:

```sh
streamlit run main.py
```

Access the web interface at [http://localhost:8501](http://localhost:8501).

### Fill in the project details:

- Project Title  
- Location  
- Duration  
- Budget (optional)  
- Project Description  

Click **"Generate Tender"** to create the document.

Review the generated sections and download in your preferred format.

## Project Structure

```
ai-tender-generator/
├── app/
│   ├── __init__.py         # Package initialization
│   ├── generator.py        # Tender generation logic
│   ├── utils.py            # Utility functions
│   └── config.py           # Configuration settings
├── static/
│   └── styles.css          # Custom CSS for Streamlit
├── .streamlit/
│   └── secrets.toml        # API keys and secrets
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── main.py                 # Main Streamlit application
```

## Tender Sections

The generator creates the following standard sections:

1. NOTICE INVITING TENDER  
2. BRIEF INTRODUCTION  
3. INSTRUCTION TO BIDDERS  
4. SCOPE OF WORK  
5. TERMS AND CONDITIONS  
6. PRICE BID  

## Technical Details

### Components

- **Frontend:** Streamlit  
- **Language Model:** OpenAI GPT-4  
- **Vector Database:** Pinecone  
- **Embedding Model:** OpenAI Ada-002  

### Key Classes

- `TenderGenerator`: Main class handling tender generation  
- `get_embedding()`: Generates embeddings for text  
- `search_similar_sections()`: Searches for similar tender sections  
- `generate_tender_section()`: Generates individual sections  
- `generate_complete_tender()`: Orchestrates complete tender generation  

## Error Handling

The application includes comprehensive error handling for:

- API failures
- Client initialization issues
- Generation errors
- Invalid input validation

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to the branch  
5. Create a Pull Request  

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the GitHub repository or contact [your-email@example.com].

## Acknowledgments

- **OpenAI** for providing the GPT-4 API  
- **Pinecone** for vector similarity search capabilities  
- **Streamlit** for the web interface framework
