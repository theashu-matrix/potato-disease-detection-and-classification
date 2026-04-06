import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from fpdf import FPDF
from datetime import datetime
import os
import tempfile

# --- CONFIGURATION ---
# Set to False for IBM Review (Binary: Healthy vs Disease)
# Set to True for Capstone (Detailed: Specific Disease Name)
CAPSTONE_MODE = True

# Confidence Threshold (Reject images below this %)
CONFIDENCE_THRESHOLD = 60

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AgroScan AI | Potato Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #F5F7F3;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    h1 { color: #1B5E20; }
    h2, h3 { color: #388E3C; }
    </style>
""", unsafe_allow_html=True)

# --- EXTENDED EXPERT DISEASE DATA (Indian Context) ---
extended_disease_info = {
    "Bacteria": {
        "disease_name": "Bacterial Wilt / Brown Rot (bacteriosis)",
        "disease_type": "Bacterial Infection",
        "summary": {
            "what_is_it": "A serious bacterial infection that blocks water flow in the plant.",
            "why_it_occurs": "Caused by bacteria in soil or infected seeds. High moisture/heat makes it worse.",
            "how_it_spreads": "Spreads through irrigation water, farm tools, and root contact.",
            "immediate_action": "Immediately remove and burn the infected plant. Do not compost.",
            "what_to_avoid": "Avoid over-watering and do not plant potatoes in the same field next season."
        },
        "preventive_measures": {
            "cultural": [
                "Crop Rotation: Rotate with maize/wheat for 2-3 years.",
                "Use certified disease-free seed tubers.",
                "Ensure proper field drainage."
            ],
            "biological": ["Soil application of Pseudomonas fluorescens @ 2.5 kg/ha."],
            "chemical": ["Soil bleaching powder application before planting."],
            "precautions": ["Clean farm tools with dilute bleach after handling infected plants."]
        },
        "medicines": [
            {
                "medicine_name": "Bleaching Powder (Stable)",
                "active_ingredient": "Calcium Hypochlorite",
                "dosage": "15 kg/hectare",
                "application_method": "Soil drenching / Mixed with fertilizer",
                "purchase_links": ["https://www.iffcobazar.in", "Local Fertilizer Shop"]
            },
            {
                "medicine_name": "Blitox / Blue Copper",
                "active_ingredient": "Copper Oxychloride 50% WP",
                "dosage": "3g per liter water",
                "application_method": "Foliar Spray or Soil Drenching",
                "purchase_links": ["https://www.bighaat.com", "Local Agri Store"]
            }
        ],
        "disclaimer": "Bacterial wilt is hard to cure. Prevention is best. Follow local agri officer advice."
    },
    
    "Fungi": {
        "disease_name": "Early Blight (Alternaria solani)",
        "disease_type": "Fungal Infection",
        "summary": {
            "what_is_it": "A fungal disease creating target-like brown spots on leaves.",
            "why_it_occurs": "Warm temperatures (24-29°C) and alternating wet/dry weather.",
            "how_it_spreads": "Spores travel via wind, rain, or infected debris.",
            "immediate_action": "Prune affected lower leaves and spray fungicide immediately.",
            "what_to_avoid": "Avoid overhead sprinklers; use drip irrigation to keep leaves dry."
        },
        "preventive_measures": {
            "cultural": [
                "Maintain proper spacing for air circulation.",
                "Remove plant debris after harvest.",
                "Deep summer ploughing."
            ],
            "biological": ["Spray Trichoderma viride formulations early in the season."],
            "chemical": ["Seed treatment with Mancozeb before sowing."],
            "precautions": ["Spray in the morning on a clear day."]
        },
        "medicines": [
            {
                "medicine_name": "Indofil M-45 / Dithane M-45",
                "active_ingredient": "Mancozeb 75% WP",
                "dosage": "2.5g per liter water",
                "application_method": "Foliar Spray",
                "purchase_links": ["https://www.bighaat.com", "Amazon India Agri"]
            },
            {
                "medicine_name": "Amistar / Contaf",
                "active_ingredient": "Azoxystrobin or Hexaconazole",
                "dosage": "1ml per liter water",
                "application_method": "Foliar Spray (severe cases)",
                "purchase_links": ["https://www.iffcobazar.in", "Local Dealer"]
            }
        ],
        "disclaimer": "Rotate fungicides to prevent resistance. Follow label instructions."
    },

    "Virus": {
        "disease_name": "Potato Leaf Roll Virus (PLRV) / Mosaic",
        "disease_type": "Viral Infection",
        "summary": {
            "what_is_it": "Viral infection that stunts growth and rolls leaves upward.",
            "why_it_occurs": "Transmitted by small insects like aphids (greenfly).",
            "how_it_spreads": "Sucking pests transfer virus from sick to healthy plants.",
            "immediate_action": "Pull out and destroy the plant (Roguing). It cannot be cured.",
            "what_to_avoid": "Do not keep seed tubers from this crop."
        },
        "preventive_measures": {
            "cultural": ["Use certified virus-free seeds.", "Remove weeds hosting aphids."],
            "biological": ["Grow barrier crops like maize around the field."],
            "chemical": ["Control the vector (Aphids) to stop spread."],
            "precautions": ["Monitor field for aphids regularly."]
        },
        "medicines": [
            {
                "medicine_name": "Confidor / Tata Mida",
                "active_ingredient": "Imidacloprid 17.8% SL",
                "dosage": "0.5ml per liter water",
                "application_method": "Foliar Spray (Targeting Aphids)",
                "purchase_links": ["https://www.bighaat.com", "Local Agri Shop"]
            },
            {
                "medicine_name": "Actara",
                "active_ingredient": "Thiamethoxam 25% WG",
                "dosage": "0.5g per liter water",
                "application_method": "Foliar Spray",
                "purchase_links": ["https://www.iffcobazar.in", "Local Dealer"]
            }
        ],
        "disclaimer": "Viruses cannot be cured. Medicines listed kill the insect vectors (Aphids)."
    },

    "Pest": {
        "disease_name": "Potato Tuber Moth / Beetle / Aphids",
        "disease_type": "Pest Infestation",
        "summary": {
            "what_is_it": "Damage caused by insects eating leaves or sucking sap.",
            "why_it_occurs": "Favorable weather and lack of natural predators.",
            "how_it_spreads": "Insects fly or crawl from nearby infested fields.",
            "immediate_action": "Spray insecticide and set up sticky traps.",
            "what_to_avoid": "Avoid excess nitrogen fertilizer which attracts pests."
        },
        "preventive_measures": {
            "cultural": ["Earthing up to prevent moth entry.", "Install Yellow Sticky Traps."],
            "biological": ["Release Trichogramma egg parasitoids."],
            "chemical": ["Neem Oil spray (10000 ppm) as deterrent."],
            "precautions": ["Do not spray during flowering to protect bees."]
        },
        "medicines": [
            {
                "medicine_name": "Proclaim",
                "active_ingredient": "Emamectin Benzoate 5% SG",
                "dosage": "0.5g per liter water",
                "application_method": "Foliar Spray",
                "purchase_links": ["https://www.bighaat.com", "Local Dealer"]
            },
            {
                "medicine_name": "Neem Oil",
                "active_ingredient": "Azadirachtin",
                "dosage": "3-5ml per liter water",
                "application_method": "Foliar Spray (Eco-friendly)",
                "purchase_links": ["https://www.iffcobazar.in", "Amazon India"]
            }
        ],
        "disclaimer": "Use safety gear (gloves/mask) while spraying chemicals."
    },

    "Healthy": {
        "disease_name": "Healthy Potato Plant",
        "disease_type": "No Disease Detected",
        "summary": {
            "what_is_it": "The plant is growing normally.",
            "why_it_occurs": "Good management and favorable conditions.",
            "how_it_spreads": "N/A",
            "immediate_action": "Continue regular care.",
            "what_to_avoid": "Do not become complacent; keep monitoring."
        },
        "preventive_measures": {
            "cultural": ["Maintain irrigation schedule.", "Weeding."],
            "biological": [],
            "chemical": [],
            "precautions": []
        },
        "medicines": [],
        "disclaimer": "Great job! Keep monitoring your field."
    }
}

# --- 1. LOAD RESOURCES (Cached) ---
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

@st.cache_resource
def load_model():
    model = models.efficientnet_b2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, len(class_names))
    )
    # Ensure map_location is set to cpu for compatibility
    model.load_state_dict(torch.load('best_potato_model_final.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 2. HELPER FUNCTIONS ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(260 + 32),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image, model):
    tensor = process_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)
        confidence = top_prob.item() * 100
        predicted_class = class_names[top_class.item()]
    return predicted_class, confidence

def create_pdf_report(prediction, confidence, uploaded_file, info):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    image_temp_path = None

    def add_section_title(title):
        pdf.ln(2)
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(27, 94, 32)
        pdf.cell(0, 8, txt=title, ln=True)
        pdf.set_text_color(0, 0, 0)

    def add_body_text(text, indent=0):
        content = text if text else "N/A"
        x_position = pdf.l_margin + indent
        width = page_width - indent
        pdf.set_x(x_position)
        pdf.multi_cell(width, 7, txt=content)

    def add_label_value(label, value):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 7, txt=f"{label}:", ln=True)
        pdf.set_font("Arial", size=11)
        add_body_text(value, indent=4)

    def add_bullet_list(items):
        if not items:
            add_body_text("- N/A")
            return
        pdf.set_font("Arial", size=11)
        for item in items:
            add_body_text(f"- {item}")

    try:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt="AgroScan AI - Analysis Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        if uploaded_file is not None:
            pdf.cell(0, 8, txt=f"Uploaded File: {uploaded_file.name}", ln=True, align='C')
        pdf.ln(4)

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_image:
                temp_image.write(uploaded_file.getvalue())
                image_temp_path = temp_image.name

            try:
                image_width = min(page_width, 90)
                pdf.image(image_temp_path, x=pdf.l_margin + ((page_width - image_width) / 2), w=image_width)
                pdf.ln(4)
            except RuntimeError:
                pass

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt=f"Diagnosis Result: {prediction}", ln=True)
        pdf.cell(0, 8, txt=f"AI Confidence: {confidence:.2f}%", ln=True)

        disease_name = info.get("disease_name") if info else None
        disease_type = info.get("disease_type") if info else None
        if disease_name:
            pdf.cell(0, 8, txt=f"Expert Classification: {disease_name}", ln=True)
        if disease_type:
            pdf.cell(0, 8, txt=f"Disease Type: {disease_type}", ln=True)

        if prediction != "Healthy":
            pdf.set_text_color(194, 24, 7)
            pdf.cell(0, 8, txt="Status: IMMEDIATE ATTENTION REQUIRED", ln=True)
        else:
            pdf.set_text_color(46, 125, 50)
            pdf.cell(0, 8, txt="Status: CROP IS HEALTHY", ln=True)
        pdf.set_text_color(0, 0, 0)

        if info:
            summary = info.get("summary", {})
            preventive_measures = info.get("preventive_measures", {})
            medicines = info.get("medicines", [])

            add_section_title("Summary")
            add_label_value("What is it", summary.get("what_is_it"))
            add_label_value("Why it occurs", summary.get("why_it_occurs"))
            add_label_value("How it spreads", summary.get("how_it_spreads"))

            add_section_title("Immediate Actions")
            add_body_text(summary.get("immediate_action"))

            add_section_title("Avoid")
            add_body_text(summary.get("what_to_avoid"))

            add_section_title("Medicines")
            if medicines:
                for index, med in enumerate(medicines, start=1):
                    pdf.set_font("Arial", 'B', 11)
                    pdf.cell(0, 7, txt=f"Medicine {index}", ln=True)
                    pdf.set_font("Arial", size=11)
                    add_body_text(f"Name: {med.get('medicine_name', 'N/A')}")
                    add_body_text(f"Ingredient: {med.get('active_ingredient', 'N/A')}")
                    add_body_text(f"Dosage: {med.get('dosage', 'N/A')}")
                    add_body_text(f"Method: {med.get('application_method', 'N/A')}")
                    purchase_links = med.get("purchase_links", [])
                    if purchase_links:
                        add_body_text(f"Availability: {', '.join(purchase_links)}")
                    pdf.ln(1)
            else:
                add_body_text("No medicine recommendation listed.")

            add_section_title("Preventive Measures")
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, txt="Cultural", ln=True)
            add_bullet_list(preventive_measures.get("cultural", []))

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, txt="Biological", ln=True)
            add_bullet_list(preventive_measures.get("biological", []))

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 7, txt="Chemical", ln=True)
            add_bullet_list(preventive_measures.get("chemical", []))

            precautions = preventive_measures.get("precautions", [])
            if precautions:
                pdf.set_font("Arial", 'B', 11)
                pdf.cell(0, 7, txt="Precautions", ln=True)
                add_bullet_list(precautions)

            add_section_title("Disclaimer")
            pdf.set_font("Arial", 'I', 10)
            add_body_text(info.get("disclaimer"))
        else:
            add_section_title("Disclaimer")
            pdf.set_font("Arial", 'I', 10)
            add_body_text("Detailed expert data was not available for this prediction.")

        return pdf.output(dest="S").encode("latin-1")
    finally:
        if image_temp_path and os.path.exists(image_temp_path):
            os.remove(image_temp_path)

# --- 3. MAIN APP LAYOUT ---
def main():
    # SIDEBAR NAVIGATION
    st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.sidebar.title("AgroScan AI")
    st.sidebar.subheader("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Scanner Dashboard", "About Project", "System Health"])
    
    # --- SECTION 1: SCANNER DASHBOARD ---
    if app_mode == "Scanner Dashboard":
        # Horizontal Banner with Correct Parameter
        st.image("image/my_background.jpg", width="stretch") 
        
        st.title("🌿 Potato Disease Diagnostics")
        st.markdown("Upload a leaf image to generate a real-time health analysis report.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Subject Image', use_column_width=True, channels="RGB")
        
        with col2:
            if uploaded_file and st.button('Run Diagnostics'):
                with st.spinner('Processing image with EfficientNet-B2...'):
                    prediction, confidence = predict(image, model)
                    
                    # --- SAFETY FILTER ---
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ **Image Rejected**")
                        st.error(f"Confidence ({confidence:.2f}%) is below safety threshold.")
                        st.info("The system could not definitively identify a potato leaf. Please upload a clearer image.")
                    
                    else:
                        # LOGIC SWITCH (IBM vs Capstone)
                        display_pred = prediction
                        display_status = "Healthy" if prediction == "Healthy" else "Infected"
                        
                        # IBM Mode: Mask specific name if infected
                        if not CAPSTONE_MODE and display_status == "Infected":
                             display_pred = "Pathogen Detected"
                        
                        # --- RESULTS DISPLAY ---
                        st.subheader("Analysis Results")
                        metric_color = "green" if display_status == "Healthy" else "red"
                        st.markdown(f"""
                        <div style="padding: 20px; background-color: white; border-radius: 10px; border-left: 5px solid {metric_color}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h3 style="margin:0; color:gray;">Status</h3>
                            <h1 style="margin:0; color:{metric_color};">{display_pred}</h1>
                            <p style="margin:0;">Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("###")
                        st.progress(int(confidence))
                        
                        # --- EXPERT INFORMATION DISPLAY ---
                        # Fetch Expert Data from Extended Dictionary
                        # Note: If IBM mode is on, 'display_pred' is "Pathogen Detected", 
                        # so we must use the original 'prediction' to look up data.
                        info = extended_disease_info.get(prediction, None)
                        
                        if info:
                            st.success(f"👨‍🌾 **Farmer's Summary:** {info['summary']['immediate_action']}")
                            
                            with st.expander("📖 Read Simple Explanation"):
                                st.markdown(f"""
                                * **What is it?** {info['summary']['what_is_it']}
                                * **Why it happens:** {info['summary']['why_it_occurs']}
                                * **How it spreads:** {info['summary']['how_it_spreads']}
                                * **⚠️ AVOID:** {info['summary']['what_to_avoid']}
                                """)
                            
                            if prediction != "Healthy":
                                tab1, tab2 = st.tabs(["💊 Medicines (Indian Market)", "🛡️ Prevention & Control"])
                                
                                with tab1:
                                    st.warning(f"⚠️ **Disclaimer:** {info['disclaimer']}")
                                    for med in info['medicines']:
                                        st.markdown(f"""
                                        ---
                                        **Name:** {med['medicine_name']}  
                                        **Chemical:** *{med['active_ingredient']}* **Dosage:** `{med['dosage']}`  
                                        **Method:** {med['application_method']}  
                                        [🛒 Check Availability]({med['purchase_links'][0]})
                                        """)
                                
                                with tab2:
                                    st.markdown("**🌱 Cultural Methods:**")
                                    for item in info['preventive_measures']['cultural']:
                                        st.markdown(f"- {item}")
                                    st.markdown("**🦠 Biological/Organic:**")
                                    for item in info['preventive_measures']['biological']:
                                        st.markdown(f"- {item}")
                                    st.markdown("**🧪 Chemical Prevention:**")
                                    for item in info['preventive_measures']['chemical']:
                                        st.markdown(f"- {item}")
                        
                        # --- REPORT DOWNLOAD ---
                        st.markdown("---")
                        pdf_bytes = create_pdf_report(display_pred, confidence, uploaded_file, info)
                        st.download_button(
                            label="📄 Download Official PDF Report",
                            data=pdf_bytes,
                            file_name="AgroScan_Report.pdf",
                            mime="application/pdf"
                        )

    # --- SECTION 2: ABOUT PROJECT ---
    elif app_mode == "About Project":
        st.title("About AgroScan AI")
        st.write("""
        **AgroScan AI** is a final-year Capstone project designed to bring precision agriculture to potato farming.
        
        ### 🧠 The Technology
        * **Architecture:** EfficientNet-B2 (Transfer Learning)
        * **Training:** PyTorch on Google Colab (Tesla T4 GPU)
        * **Accuracy:** 73.8% (Validated on PlantVillage Dataset)
        * **Features:** Multi-stage fine-tuning & Random Erasing Augmentation
        * **Safety:** Confidence Thresholding to prevent false positives.
        
        ### 🎯 Problem Statement
        Early Blight and Late Blight are major threats to global food security. 
        Manual identification is slow and error-prone. This tool automates the process using Computer Vision.
        """)

    # --- SECTION 3: SYSTEM HEALTH ---
    elif app_mode == "System Health":
        st.title("System Status")
        st.metric(label="Model Status", value="Active", delta="Loaded")
        st.metric(label="Backend Device", value="CPU", delta="Ready")
        st.json({
            "model_version": "v1.0.4 (Random Erasing)",
            "classes_loaded": len(class_names),
            "threshold": f"{CONFIDENCE_THRESHOLD}%"
        })

if __name__ == "__main__":
    main()
