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
        """Get embedding for text using OpenAI API"""
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
        """Search for similar tender sections"""
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
        """Generate a specific section of the tender using GPT-4"""
        try:
            # Create context from similar sections
            context = "\n\n".join([
                f"Example {i+1}:\n{match['metadata']['content']}"
                for i, match in enumerate(similar_sections)
            ])

            # Create prompt for GPT-4
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
        """Generate complete tender document"""
        tender_sections = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for idx, section in enumerate(self.sections):
                status_text.text(f"Generating {section}...")

                # Search for similar sections
                search_query = f"{section} {project_details['title']} {project_details['description']}"
                similar_sections = self.search_similar_sections(search_query)

                # Generate section content
                section_content = self.generate_tender_section(
                    section,
                    project_details,
                    similar_sections
                )
                tender_sections[section] = section_content

                # Update progress
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
    """Initialize OpenAI and Pinecone clients"""
    try:
        openai_client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            http_client=httpx.Client(
                timeout=60.0,
                follow_redirects=True
            )
        )

        # Initialize Pinecone using the current method
        pc = pinecone.Pinecone(
            api_key=st.secrets["PINECONE_API_KEY"]
        )

        # Get the index
        index = pc.Index("tender-documents")

        return openai_client, index
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        raise e

def main():
    st.set_page_config(page_title="AI Tender Generator", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
        .tender-section {
            margin: 1rem 0;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .stProgress > div > div > div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("AI Tender Generator")
    st.markdown("""
    This application generates professional tender documents based on your requirements,
    learning from existing tender documents in our database.
    """)

    try:
        # Initialize clients
        openai_client, pinecone_index = init_clients()
        tender_generator = TenderGenerator(openai_client, pinecone_index)

        # Project Details Form
        with st.form("tender_details"):
            st.subheader("Project Details")
            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input("Project Title*")
                location = st.text_input("Project Location*")
                duration = st.number_input("Project Duration (months)*",
                                         min_value=1, value=12)

            with col2:
                budget = st.text_input("Project Budget")
                description = st.text_area("Project Description*",
                                         help="Provide a detailed description of the project")

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
                    # Generate tender
                    tender_sections = tender_generator.generate_complete_tender(project_details)

                    # Display generated tender
                    st.success("Tender document generated successfully!")

                    # Create tabs for different sections
                    tabs = st.tabs(tender_generator.sections)

                    for tab, section_name in zip(tabs, tender_generator.sections):
                        with tab:
                            st.markdown(f"### {section_name}")
                            st.markdown(tender_sections[section_name])

                    # Prepare download
                    tender_doc = "\n\n".join([
                        f"# {section}\n\n{content}"
                        for section, content in tender_sections.items()
                    ])

                    # Add download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download as Text",
                            data=tender_doc,
                            file_name=f"tender_{title.lower().replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                    with col2:
                        # Convert to JSON for structured data
                        tender_json = json.dumps(tender_sections, indent=2)
                        st.download_button(
                            label="Download as JSON",
                            data=tender_json,
                            file_name=f"tender_{title.lower().replace(' ', '_')}.json",
                            mime="application/json"
                        )

                except Exception as e:
                    st.error(f"An error occurred during tender generation: {str(e)}")

        # Sidebar with information
        st.sidebar.title("About")
        st.sidebar.info("""
        This tender generator uses AI to create professional tender documents based on your
        project requirements. It learns from a database of existing tenders to ensure
        accuracy and compliance with standard formats.
        """)

        # Display statistics in sidebar
        st.sidebar.title("Database Statistics")
        try:
            stats = pinecone_index.describe_index_stats()
            st.sidebar.metric("Total Documents", stats['total_vector_count'])
            st.sidebar.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            st.sidebar.error(f"Error fetching statistics: {str(e)}")

    except Exception as e:
        st.error(f"Application initialization error: {str(e)}")

if __name__ == "__main__":
    main()
