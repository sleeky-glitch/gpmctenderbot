import streamlit as st
import pinecone
from openai import OpenAI
import httpx
from typing import List, Dict, Any
import json
from datetime import datetime

class TenderGenerator:
    def __init__(self, openai_client, pinecone_index):
        self.client = openai_client
        self.index = pinecone_index
        self.sections = [
            "NOTICE INVITING TENDER",
            "BRIEF INTRODUCTION",
            "INSTRUCTION TO BIDDERS",
            "SCOPE OF WORK",
            "TERMS AND CONDITIONS",
            "PRICE BID"
        ]

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise e

    def search_similar_sections(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results['matches']
        except Exception as e:
            st.error(f"Error searching similar sections: {str(e)}")
            raise e

    def generate_tender_section(self, section_name: str, project_details: Dict[str, str],
                              similar_sections: List[Dict[str, Any]]) -> str:
        try:
            context = "\n\n".join([
                f"Example {i+1}:\n{match['metadata']['content']}"
                for i, match in enumerate(similar_sections)
            ])

            prompt = f"""You are an expert tender document writer. Generate the {section_name} section
            for a new tender based on the following project details and example sections.

            Project Details:
            Title: {project_details['title']}
            Location: {project_details['location']}
            Duration: {project_details['duration']} months
            Budget: {project_details['budget']}
            Description: {project_details['description']}

            Similar Examples from Other Tenders:
            {context}

            Please generate a professional and detailed {section_name} section that follows
            the style and format of the examples while being specific to this project.
            The content should be practical, clear, and legally sound."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert tender document generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating tender section: {str(e)}")
            raise e

    def generate_complete_tender(self, project_details: Dict[str, str]) -> Dict[str, str]:
        tender_sections = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for idx, section in enumerate(self.sections):
                status_text.text(f"Generating {section}...")
                search_query = f"{section} {project_details['title']} {project_details['description']}"
                similar_sections = self.search_similar_sections(search_query)
                section_content = self.generate_tender_section(
                    section,
                    project_details,
                    similar_sections
                )
                tender_sections[section] = section_content
                progress = (idx + 1) / len(self.sections)
                progress_bar.progress(progress)

            status_text.text("Tender generation completed!")
            return tender_sections
        except Exception as e:
            st.error(f"Error generating complete tender: {str(e)}")
            raise e
        finally:
            progress_bar.empty()
            status_text.empty()

@st.cache_resource
def init_clients():
    try:
        openai_client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            http_client=httpx.Client(
                timeout=60.0,
                follow_redirects=True
            )
        )
        pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index("tender-documents")
        return openai_client, index
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        raise e

def main():
    st.set_page_config(
        page_title="GMDC Tender Generator",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS with improved styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
            background-color: #0066cc;
            color: white;
        }
        .tender-section {
            margin: 1rem 0;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #ffffff;
        }
        .stProgress > div > div > div {
            background-color: #0066cc;
        }
        .header-container {
            display: flex;
            align-items: center;
            padding: 10px 20px;
            background-color: #f8f9fa;
            border-bottom: 2px solid #eee;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .logo-image {
            width: 100px;
            height: auto;
            object-fit: contain;
            margin-right: 20px;
        }
        .header-text {
            flex-grow: 1;
        }
        .header-text h1 {
            margin: 0;
            color: #333;
            font-size: 2em;
        }
        .subheader {
            color: #666;
            font-size: 1.1em;
            margin: 5px 0 0 0;
        }
        .stTab {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with logo and text
    st.markdown(f"""
        <div class="header-container">
            <img class="logo-image"
                 src="https://upload.wikimedia.org/wikipedia/en/d/df/GujaratMineralDevelopmentCorporationLogo.jpg"
                 alt="GMDC Logo">
            <div class="header-text">
                <h1>GMDC Tender Generator</h1>
                <p class="subheader">Gujarat Mineral Development Corporation Limited</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


    try:
        openai_client, pinecone_index = init_clients()
        tender_generator = TenderGenerator(openai_client, pinecone_index)

        # Project Details Form
        with st.form("tender_details"):
            st.subheader("Project Details")
            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input("Project Title*")
                location = st.text_input("Project Location*")
                duration = st.number_input(
                    "Project Duration (months)*",
                    min_value=1,
                    value=12
                )

            with col2:
                budget = st.text_input("Project Budget")
                description = st.text_area(
                    "Project Description*",
                    help="Provide a detailed description of the project"
                )

            st.markdown("*Required fields")
            generate_button = st.form_submit_button("Generate Tender")

        if generate_button:
            if not all([title, location, description]):
                st.error("Please fill in all required fields.")
                return

            project_details = {
                "title": title,
                "location": location,
                "duration": duration,
                "budget": budget or "Not specified",
                "description": description
            }

            with st.spinner("Generating tender document... This may take a few minutes."):
                try:
                    tender_sections = tender_generator.generate_complete_tender(project_details)
                    st.success("Tender document generated successfully!")

                    # Display sections in tabs
                    tabs = st.tabs(tender_generator.sections)
                    for tab, section_name in zip(tabs, tender_generator.sections):
                        with tab:
                            st.markdown(f"### {section_name}")
                            st.markdown(tender_sections[section_name])

                    # Prepare downloads
                    tender_doc = "\n\n".join([
                        f"# {section}\n\n{content}"
                        for section, content in tender_sections.items()
                    ])
                    tender_json = json.dumps(tender_sections, indent=2)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as Text",
                            data=tender_doc,
                            file_name=f"tender_{title.lower().replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                    with col2:
                        st.download_button(
                            label="Download as JSON",
                            data=tender_json,
                            file_name=f"tender_{title.lower().replace(' ', '_')}.json",
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"An error occurred during tender generation: {str(e)}")

    except Exception as e:
        st.error(f"Application initialization error: {str(e)}")

if __name__ == "__main__":
    main()
