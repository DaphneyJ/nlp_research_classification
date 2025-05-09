import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set page title 
st.set_page_config(page_title="Research Bio Classifier", layout="wide")

# Header
st.title("Research Classification Tool by Collaborative Real Estate")
st.markdown("Paste faculty research bio to classify primary research area")

# input text area
text_area = st.text_area("Paste research bio here:", height=200)

@st.cache_resource
def load_model():
    """Load the sentence transformer model - cached to avoid reloading"""
    return SentenceTransformer('all-MiniLM-L6-v2') # all-mpnet-base-v2 model


# Load model
model = load_model()

# define HERD research topics
categories = [
    # "Computer and information sciences",
    # "Atmospheric science and meteorology",
    # "Geological and earth sciences",
    # "ocean sciences and marine sciences",
    # "Agricultural sciences",
    # "Biological and biomedical sciences",
    # "Health sciences",
    # "Natural resources and conservation",
    # "Mathematics and statistics",
    # "Astronomy and astrophysics",
    # "Chemistry",
    # "Materials science",
    # "Physics",
    # "Psychology",
    # "Economics",
    # "Political science and government",
    # "Sociology, demography, and population studies",
    # "Sociology",
    # "Aerospace, aeronautical, and astronautical engineering",
    # "Bioengineering and biomedical engineering",
    # "Chemical engineering",
    # "Civil engineering",
    # "Electrical, electronic, and communications engineering",
    # "Industrial and manufacturing engineering",
    # "Mechanical engineering",
    # "Metallurgical and materials engineering",
    # "Business management and business administration",
    # "Education",
    # "Humanities",
    # "Law",
    # "Social work",
    # "Visual and performing arts",
    "Computer and information sciences, general",
    "Artificial intelligence and robotics",
    "Information technology",
    "Informatics",
    "Computer science, other",
    "Computer programming/programmer, general",
    "Computer programming special applications",
    "Computer Programming, Vendor/Product Certification",
    "Computer programming, other",
    "Data processing and data processing technology/technician",
    "Information science/studies",
    "Computer science",
    "Web page, digital/multimedia, and information resources design",
    "Data modeling/warehousing and database administration",
    "Computer graphics",
    "Modeling, virtual environments and simulation",
    "Computer software and media applications, other",
    "Computer systems networking and telecommunications",
    "System administration/administrator",
    "System, networking, and LAN/WAN management/manager",
    "Computer and information systems security",
    "Web/multimedia management and Webmaster",
    "Information technology project management",
    "Computer support specialist",
    "Computer/information technology services administration and management, other",

    "Atmospheric sciences and meteorology, general",
    "Atmospheric Chemistry and Climatology",
    "Atmospheric physics and dynamics",
    "Meteorology",

    "Geology",
    "Geochemistry",
    "Paleontology",
    "Geophysics and seismology",
    "Mineralogy and petrology",
    "Stratigraphy and sedimentation",
    "Geomorphology and glacial geology",
    "Geological and earth sciences, general",

    "Marine biology and biological oceanography",
    "Marine sciences",
    "Hydrology and water resources science",
    "Oceanography, chemical and physical",

    "Agricultural economics",
    "Natural resources/environmental economics",
    "Agricultural animal breeding",
    "Animal nutrition",
    "Animal science",
    "Agronomy and crop science",
    "Agricultural and horticultural plant breeding",
    "Environmental science",
    "Plant pathology/phytopathology ",
    "Plant sciences, other",
    "Food science",
    "Food science and technology",
    "Soil chemistry/microbiology/physics",
    "Soil sciences",
    "Horticulture science",
    "Fishing and fisheries sciences/management",
    "Forest sciences and biology",
    "Forest/resources management",
    "Wood science and pulp/paper technology",
    "Natural resources/conservation ",
    "Forestry and related sciences, other",
    "Wildlife/range management",
    "Environmental science",
    "Agriculture, general",
    "Agricultural science, other",
    "Natural resource/environmental policy",

    "Biological sciences",
    "Biochemistry",
    "Bioinformatics",
    "Biomathematics",
    "Biomedical sciences",
    "Computational biology",
    "Biophysics",
    "Biotechnology",
    "Bacteriology",
    "Plant genetics",
    "Plant pathology/phytopathology",
    "Plant physiology",
    "Botany/plant biology",
    "Aquatic biology",
    "Anatomy",
    "Conservation biology",
    "Biometrics and biostatistics",
    "Epidemiology",
    "Cell/cellular biology and histology",
    "Evolutionary biology",
    "Ecology",
    "Developmental biology/embryology",
    "Endocrinology",
    "Entomology",
    "Immunology",
    "Molecular biophysics",
    "Molecular biochemistry",
    "Molecular biology",
    "Structural biology",
    "Microbiology",
    "Cancer biology",
    "Molecular medicine",
    "Neuroscience",
    "Nutrition sciences",
    "Radiation biology/radiobiology",
    "Parasitology",
    "Environmental toxicology",
    "Virologye",
    "Toxicology",
    "Genetics, other",
    "Genome sciences/genomics",
    "Genetics/genomics, human and animal",
    "Pathology, human and animal",
    "Pharmacology, human and animal",
    "Physiology, human and animal",
    "Wildlife biology",
    "Zoology, other",
    "Reproductive biology",
    "Cardiovascular science",
    "Exercise physiology",
    "Vision science/physiological optics",
    "Biology/biomedical sciences, general",
    "Biology/biomedical sciences, other",

    "Oral biology/oral pathology",
    "Environmental health",
    "Environmental toxicology",
    "Health systems/services administration",
    "Health services research",
    "Public health",
    "Epidemiology",
    "Kinesiology/exercise science",
    "Gerontology",
    "Nursing science",
    "Medicinal/pharmaceutical sciences",
    "Rehabilitation/therapeutic services",
    "Veterinary sciences",
    "Health and behavioral",
    "Health sciences, general",
    "Health sciences, other",
    "Medical physics/radiological sciencee",
    "Audiology/audiologist and hearing sciences",
    "Speech-language pathology/pathologist",
    "Communication disorders",
    "Dental clinical sciences, general",
    "Oral/maxillofacial surgery",
    "Pediatric dentistry/pedodontics",
    "Hospital and health care facilities administration/management",
    "Anesthesiology",
    "Gene/genetic therapy",
    "Pharmaceutics and drug design",
    "Pharmacy administration and pharmacy policy and regulatory affairs",
    "Pharmacoeconomics/pharmaceutical economics",
    "Clinical and industrial drug development",
    "Maternal and child health",
    "Veterinary sciences",
    "Dietetics/dietitian",
    "Holistic health",
    "Herbalism",
    "Psychiatric/mental health",

    "Mathematics, general",
    "Analysis and functional analysis",
    "Mathematics, other",
    "Applied mathematics",
    "Computational mathematics",
    "Computational and applied mathematics",
    "Financial mathematics",
    "Mathematical biology",
    "Applied mathematics, other",
    "Statistics, general",
    "Mathematical statistics and probability",
    "Mathematics and statistics",
    "Operations research",
    "Mathematics and statistics, other",
    "Business statistics",
    "Actuarial science",
    "Geometry/geometric analysis",
    "Number theory",
    "Topology/foundations",

    "Astronomy",
    "Astrophysics",
    "Planetary astronomy and science",
    "Astronomy and astrophysics, other",

    "Chemistry, general",
    "Analytical chemistry",
    "Inorganic chemistry",
    "Organic chemistry",
    "Physical and theoretical chemistry",
    "Polymer chemistry",
    "Chemical physics",
    "Environmental chemistry",
    "Forensic chemistry",
    "Theoretical chemistry",
    "Chemistry, other",
    "Medicinal and pharmaceutical chemistry",

    "Physics, general",
    "Atomic/molecular/chemical physics",
    "Elementary particle physics",
    "Nuclear physics",
    "Optics/optical sciences",
    "Solid state and low-temperature physics",
    "Acoustics",
    "Theoretical and mathematical physics",
    "Physics, other",
    "Polymer physics",
    "Plasma/fusion physics",
    "Biophysics",

    "Applied physics",
    "Physical sciences",
    "Materials chemistry",
    "Physical sciences, other",

    "Anthropology",
    "Economics",
    "Political science",

    "Psychology, general",
    "Cognitive psychology and psycholinguistics",
    "Comparative psychology",
    "Developmental and child psychology",
    "Experimental psychology",
    "Personality psychology",
    "Physiological psychology/psychobiology",
    "Social psychology",
    "Psychometrics and quantitative psychology",
    "Psychopharmacology",
    "Research and experimental psychology, other",
    "Clinical psychology",
    "Community psychology",
    "Counseling psychology",
    "Industrial and organizational psychology",
    "Clinical child psychology",
    "Environmental psychology",
    "Geropsychology",
    "Health/medical psychology",
    "Family psychology",
    "Forensic psychology",
    "Applied psychology",
    "Applied behavior analysis",
    "Clinical, counseling and applied psychology, other",
    "Psychology, other",
    "Psychoanalysis and psychotherapy",

    "Sociology",
    "Sociology and anthropology",
    "Rural sociology",

    "Health policy analysis",
    "Gender and women's studies",
    "Area/ethnic/cultural/gender studies",
    "Criminal justice and corrections",
    "Criminology",
    "Demography/population studies",
    "Geography",
    "International relations/affairs",
    "Applied linguistics",
    "Linguistics",
    "Public policy analysis",
    "Gerontology ",
    "Statistics",
    "Urban affairs/studies",
    "Urban/city, community, and regional planning",
    "Social sciences, general",
    "Social sciences, other",
    "American/U.S. studies",


    "Aerospace, aeronautical, and astronautical engineering",
    "Bioengineering and biomedical engineering",
    "Chemical engineering",
    "Chemical and biomolecular engineering",

    "Civil engineering",
    "Geotechnical engineering",
    "Structural engineering",
    "Transportation and highway engineering",
    "Water resources engineering",

    "Electrical and electronics engineering",
    "Laser and optical engineering",
    "Telecommunications engineering",

    "Industrial engineering",
    "Manufacturing engineering",
    "Engineering/industrial management",

    "Materials engineering",
    "Materials science",
    "Mechanical engineering",

    "Agricultural/biological engineering and bioengineering",
    "Architectural engineering",
    "Ceramic sciences and engineering",
    "Computer engineering, general",
    "Computer hardware engineering",
    "Computer software engineering",
    "Computer engineering, other",
    "Engineering mechanics",
    "Engineering physics",
    "Engineering science",
    "Environmental/environmental health engineering",
    "Metallurgical engineering",
    "Mining and mineral engineering",
    "Naval architecture and marine engineering",
    "Nuclear engineering",
    "Ocean engineering",
    "Petroleum engineering",
    "Systems engineering",
    "Textile sciences and engineering",
    "Polymer/plastics engineering",
    "Construction engineering",
    "Forest engineering",
    "Operations research",
    "Surveying engineering",
    "Geological/geophysical engineering",
    "Paper science and engineering",
    "Electromechanical engineering",
    "Mechatronics, robotics, and automation engineering",
    "Biochemical engineering",
    "Engineering chemistry",
    "Biological/biosystems engineering",
    "Engineering, other",
    "Cartography",

    "Education",
    "Humanities and arts" 
]



# classify faculty research based on semantic similarity
def classify_text(text, top_n=5):
    """Classify text and return top n results"""
    if not text.strip():
        return []

    # generate embeddings for each category
    category_embeddings = model.encode(categories)

    # generate embedding for the research text
    text_embedding = model.encode([text])[0]

    # calculate similarity to each category
    similarities = cosine_similarity([text_embedding], category_embeddings)[0]

    # Apply corrections for specific cases
    # rule for robotics + NLP
    if ("robot" in text.lower() and
            ("natural language" in text.lower() or "nlp" in text.lower()) and
            categories[similarities.argmax()] == "Psychology"):
        # Find the index of Computer and information sciences
        cs_idx = categories.index("Computer and information sciences")
        # Boost similarity score
        similarities[cs_idx] = max(0.95, similarities[cs_idx])


    # Get indices of top n categories
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return top categories and similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'category': categories[idx],
            'confidence': similarities[idx],
            'confidence_pct': f"{similarities[idx]:.1%}"
        })

    return results


# Create main interface
if st.button("Classify Research"):
    if text_area.strip():
        with st.spinner("Analyzing research bio..."):
            # Get classifications
            results = classify_text(text_area, top_n=5)

            if results:
                # display results
                st.subheader("Classification Results:")

                # primary match 
                st.markdown(f"### Primary Match: {results[0]['category']}")
                st.progress(float(results[0]['confidence']))
                st.markdown(f" {results[0]['confidence_pct']}")

                # Secondary matches if available
                if len(results) > 1:
                    st.markdown("### Secondary Matches:")
                    for result in results[1:]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"{result['category']}")
                            st.progress(float(result['confidence']))
                        with col2:
                            st.markdown(f" {result['confidence_pct']}")
            else:
                st.error("Unable to classify. Please provide more text.")
    else:
        st.warning("Please enter some text to classify.")

# Add some explanation text
st.markdown("---")
st.markdown("""
### How it works
This tool classifies reserach bios and matches them with the most relevant NSF HERD listed research topics. 
The model considers semantic meaning of the text not just keyword matches.
""")

