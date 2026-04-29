"""
generate_sample_pdf.py
----------------------
Generates a multi-page sample PDF on "Artificial Intelligence in Healthcare"
for use in testing the PDF-Constrained Conversational Agent.

Run: python generate_sample_pdf.py
Output: data/sample_ai_healthcare.pdf
"""

from pathlib import Path
from fpdf import FPDF

Path("data").mkdir(exist_ok=True)


def _safe(text: str) -> str:
    """Replace non-latin-1 characters with ASCII equivalents for fpdf2/Helvetica compatibility."""
    replacements = {
        "\u2014": "--",    # em dash --
        "\u2013": "-",     # en dash
        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote
        "\u201c": '"',     # left double quote
        "\u201d": '"',     # right double quote
        "\u2026": "...",   # ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


CONTENT = [
    {
        "title": "Artificial Intelligence in Healthcare: A Comprehensive Overview",
        "subtitle": "Transforming Medicine Through Intelligent Systems",
        "body": None,
        "is_cover": True,
    },
    {
        "title": "1. Introduction",
        "body": (
            "Artificial Intelligence (AI) has emerged as a transformative force in healthcare, "
            "offering unprecedented capabilities in disease diagnosis, treatment planning, drug "
            "discovery, and patient monitoring. Unlike traditional software systems that follow "
            "explicit rules, AI systems learn patterns from vast amounts of medical data, enabling "
            "them to make predictions and recommendations that were previously impossible.\n\n"
            "The global AI in healthcare market was valued at approximately $11 billion in 2023 and "
            "is projected to reach $187 billion by 2030, growing at a compound annual growth rate "
            "(CAGR) of 49.6%. This explosive growth reflects the increasing recognition of AI's "
            "potential to address critical challenges in healthcare delivery, including physician "
            "shortages, rising costs, diagnostic errors, and the explosion of medical data.\n\n"
            "This document provides a comprehensive overview of AI applications in healthcare, "
            "examines ethical challenges, presents real-world case studies, discusses limitations, "
            "and outlines future directions for this rapidly evolving field."
        ),
    },
    {
        "title": "2. Main Applications of AI in Healthcare",
        "body": (
            "2.1 Medical Imaging and Diagnostics\n"
            "One of the most successful applications of AI in healthcare is medical image analysis. "
            "Deep learning algorithms, particularly convolutional neural networks (CNNs), can analyze "
            "radiology images (X-rays, CT scans, MRIs) with accuracy comparable to or exceeding that "
            "of experienced radiologists. Google's DeepMind has developed AI systems that detect over "
            "50 eye diseases from retinal scans with 94% accuracy. Similarly, AI tools for mammography "
            "screening have demonstrated a 5.7% reduction in false positives and 9.4% reduction in "
            "false negatives compared to human radiologists.\n\n"
            "2.2 Predictive Analytics and Early Warning Systems\n"
            "AI-powered predictive analytics systems analyze patient data to forecast disease "
            "progression, identify high-risk patients, and trigger early interventions. Sepsis "
            "prediction models trained on electronic health records (EHR) data have reduced "
            "mortality by identifying at-risk patients up to 12 hours before clinical deterioration. "
            "Hospital readmission prediction algorithms help care teams identify patients who may "
            "need additional support after discharge, reducing costly readmissions by 20-30%.\n\n"
            "2.3 Natural Language Processing in Clinical Documentation\n"
            "Clinical documentation is a significant burden on healthcare providers. NLP-powered "
            "systems can automatically extract structured information from unstructured clinical notes, "
            "generate clinical summaries, and enable voice-to-text documentation. These systems "
            "reduce administrative burden by an estimated 2-3 hours per physician per day."
        ),
    },
    {
        "title": "3. Role of Machine Learning in Drug Discovery",
        "body": (
            "Machine learning (ML) has revolutionized the drug discovery pipeline, which traditionally "
            "takes 10-15 years and costs over $2.6 billion per approved drug. ML accelerates this "
            "process in several critical ways:\n\n"
            "3.1 Target Identification\n"
            "ML algorithms analyze genomic, proteomic, and clinical data to identify novel disease "
            "targets. Graph neural networks (GNNs) model protein-protein interaction networks to "
            "discover previously unknown therapeutic targets with high specificity.\n\n"
            "3.2 Molecular Design and Optimization\n"
            "Generative AI models, including variational autoencoders (VAEs) and generative adversarial "
            "networks (GANs), can design novel molecular structures with desired pharmacological "
            "properties. Insilico Medicine used an AI platform to design a novel fibrosis drug "
            "candidate in just 18 months -- compared to the typical 4-5 year timeline -- with the "
            "compound progressing to Phase I clinical trials.\n\n"
            "3.3 Clinical Trial Optimization\n"
            "ML models analyze patient records and biomarkers to identify optimal trial participants, "
            "predict dropout rates, and dynamically adjust dosing protocols. This reduces trial failure "
            "rates, which historically stand at 90% for Phase I candidates entering Phase III.\n\n"
            "3.4 Drug Repurposing\n"
            "Knowledge graph-based ML systems identify new therapeutic applications for existing "
            "approved drugs. During the COVID-19 pandemic, ML tools helped identify baricitinib and "
            "dexamethasone as effective treatments within weeks of the outbreak."
        ),
    },
    {
        "title": "4. Ethical Challenges of AI in Medical Diagnosis",
        "body": (
            "The deployment of AI in clinical settings raises profound ethical challenges that must be "
            "carefully addressed to ensure equitable and trustworthy healthcare delivery.\n\n"
            "4.1 Algorithmic Bias and Health Inequity\n"
            "AI models trained on non-representative datasets can perpetuate or amplify existing health "
            "disparities. A landmark 2019 study in Science revealed that a widely-used healthcare "
            "algorithm systematically underestimated the needs of Black patients by assigning lower risk "
            "scores compared to equally sick White patients. This occurred because the model used "
            "healthcare spending as a proxy for health needs, inadvertently encoding systemic inequities.\n\n"
            "4.2 Explainability and the Black Box Problem\n"
            "Many high-performing AI models, particularly deep learning systems, function as 'black "
            "boxes' -- they produce accurate predictions without providing interpretable reasoning. In "
            "high-stakes medical decisions, clinicians and patients demand explanations. The EU's General "
            "Data Protection Regulation (GDPR) mandates a 'right to explanation' for automated decisions, "
            "creating regulatory pressure for explainable AI (XAI) in healthcare.\n\n"
            "4.3 Privacy and Data Security\n"
            "Training effective AI models requires vast amounts of patient data, raising serious privacy "
            "concerns. Federated learning approaches, where models are trained across distributed "
            "datasets without centralizing patient data, offer a promising solution but introduce "
            "challenges in model convergence and auditability.\n\n"
            "4.4 Liability and Accountability\n"
            "When an AI system contributes to a misdiagnosis or treatment error, determining liability "
            "is legally and ethically complex. Current regulatory frameworks in most countries are "
            "inadequate for AI-specific accountability structures in medical settings."
        ),
    },
    {
        "title": "5. Case Study: AI in Diabetic Retinopathy Screening",
        "body": (
            "One of the most extensively documented real-world AI deployments in healthcare is "
            "Google's diabetic retinopathy (DR) screening program in rural India and Thailand.\n\n"
            "5.1 The Problem\n"
            "Diabetic retinopathy is a leading cause of blindness affecting approximately 103 million "
            "people worldwide. Early detection and treatment can prevent 90% of blindness cases. However, "
            "rural communities in developing nations face severe shortages of trained ophthalmologists "
            "-- with patient-to-specialist ratios as high as 200,000:1.\n\n"
            "5.2 The AI Solution\n"
            "Google developed a deep learning model trained on 128,175 retinal images graded by "
            "54 ophthalmologists. The system achieved an AUC (Area Under the Curve) of 0.991 for "
            "detecting referable DR -- exceeding the performance of the median ophthalmologist (AUC 0.981).\n\n"
            "5.3 Real-World Deployment Results\n"
            "When deployed in Aravind Eye Hospitals across India, the AI system screened over "
            "350,000 patients in its first year. Critically, the system achieved a sensitivity of 90.5% "
            "and specificity of 91.6% in real-world conditions -- maintaining performance comparable to "
            "controlled clinical trial settings.\n\n"
            "5.4 Lessons Learned\n"
            "The deployment revealed important lessons: image quality in real-world settings is "
            "significantly lower than in clinical trials, requiring robust image quality assessment "
            "modules. Integration with existing clinical workflows proved more challenging than the "
            "technical model development."
        ),
    },
    {
        "title": "6. Limitations of AI Systems in Healthcare",
        "body": (
            "Despite its transformative potential, AI in healthcare faces significant limitations "
            "that must be honestly acknowledged:\n\n"
            "6.1 Data Quality and Availability\n"
            "AI systems require large, high-quality, labeled datasets for training. In healthcare, "
            "data is often fragmented across incompatible systems, inconsistently documented, and "
            "subject to strict privacy regulations that limit data sharing. Many rare diseases lack "
            "sufficient training data for effective model development.\n\n"
            "6.2 Distribution Shift\n"
            "Models trained on data from one hospital or population often perform poorly when "
            "deployed in different settings. This 'distribution shift' problem means that extensive "
            "local validation is required before any AI system is deployed in a new healthcare "
            "environment. A chest X-ray model trained in the US may underperform in Southeast Asia "
            "due to differences in disease prevalence and imaging equipment.\n\n"
            "6.3 Limited Generalization\n"
            "Current AI systems excel at narrow, well-defined tasks but lack the holistic clinical "
            "reasoning that experienced physicians apply. An AI that accurately diagnoses diabetic "
            "retinopathy cannot simultaneously assess a patient's overall health status, emotional "
            "state, or social determinants of health.\n\n"
            "6.4 Regulatory and Integration Barriers\n"
            "FDA clearance and CE marking processes for AI medical devices are still evolving. "
            "Integration with existing Electronic Health Record (EHR) systems is technically complex "
            "and costly. Many healthcare institutions lack the IT infrastructure and AI expertise "
            "required for successful deployment.\n\n"
            "6.5 Physician Adoption and Trust\n"
            "Clinical adoption of AI tools has been slower than anticipated due to concerns about "
            "accuracy, liability, workflow disruption, and a lack of transparency. Without clinician "
            "trust and adoption, even highly accurate AI systems fail to deliver clinical value."
        ),
    },
    {
        "title": "7. Future Directions",
        "body": (
            "The future of AI in healthcare will be shaped by advances in several key areas:\n\n"
            "7.1 Multimodal AI\n"
            "Next-generation clinical AI systems will integrate data from multiple modalities -- "
            "imaging, genomics, EHR, wearables, and patient-reported outcomes -- to provide "
            "holistic, longitudinal patient assessments that surpass any single-modality system.\n\n"
            "7.2 Foundation Models for Medicine\n"
            "Large language models (LLMs) and vision-language models pre-trained on broad medical "
            "knowledge are demonstrating remarkable few-shot generalization to clinical tasks. "
            "Models like Med-PaLM 2 achieved expert-level performance on the US Medical Licensing "
            "Examination (USMLE) with a score of 86.5%.\n\n"
            "7.3 AI-Augmented Clinical Workflows\n"
            "The most promising near-term applications involve AI as a 'co-pilot' for clinicians -- "
            "flagging abnormal results, surfacing relevant clinical evidence, and automating "
            "administrative tasks -- rather than replacing clinical judgment entirely.\n\n"
            "7.4 Federated and Privacy-Preserving Learning\n"
            "Federated learning enables AI models to be trained across multiple institutions without "
            "centralizing sensitive patient data. This approach is critical for building models that "
            "generalize across diverse patient populations while complying with privacy regulations."
        ),
    },
    {
        "title": "8. Conclusion",
        "body": (
            "Artificial intelligence represents one of the most significant opportunities to improve "
            "healthcare delivery in human history. From accelerating drug discovery to enabling "
            "earlier disease detection to reducing clinician administrative burden, AI applications "
            "are demonstrating measurable clinical value across the care continuum.\n\n"
            "However, realizing this potential requires careful attention to data quality, algorithmic "
            "fairness, regulatory compliance, clinical workflow integration, and ongoing monitoring "
            "for model degradation. The most successful AI health systems will be those developed "
            "through genuine collaboration between AI researchers, clinicians, patients, and "
            "policymakers.\n\n"
            "The fundamental goal of AI in healthcare is not to replace human clinicians but to "
            "amplify their capabilities, reduce errors, and extend high-quality care to underserved "
            "populations worldwide. Achieving this vision requires unwavering commitment to both "
            "technical excellence and ethical responsibility.\n\n"
            "References: This document synthesizes findings from peer-reviewed literature including "
            "publications in Nature Medicine, The Lancet Digital Health, JAMA, and NEJM AI. "
            "All statistics cited reflect published research as of 2024."
        ),
    },
]


class HealthcarePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "AI in Healthcare - Comprehensive Overview", align="L")
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(15, 18, 195, 18)
        self.ln(2)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(160, 160, 160)
        self.cell(
            0, 10,
            "Confidential - For Evaluation Purposes Only | PDF-Constrained QA Agent Test Document",
            align="C",
        )


def generate():
    pdf = HealthcarePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=20, top=25, right=20)

    for section in CONTENT:
        pdf.add_page()

        if section.get("is_cover"):
            pdf.ln(40)
            pdf.set_font("Helvetica", "B", 24)
            pdf.set_text_color(30, 60, 120)
            pdf.multi_cell(0, 12, _safe(section["title"]), align="C")
            pdf.ln(10)
            pdf.set_font("Helvetica", "I", 14)
            pdf.set_text_color(80, 100, 140)
            pdf.multi_cell(0, 8, _safe(section["subtitle"]), align="C")
            pdf.ln(20)
            pdf.set_draw_color(30, 60, 120)
            pdf.set_line_width(1)
            pdf.line(40, pdf.get_y(), 170, pdf.get_y())
            pdf.ln(10)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(100, 100, 120)
            pdf.multi_cell(
                0, 7,
                "A production-grade technical document prepared for AI system evaluation.\n"
                "Contains 8 pages covering applications, ethics, case studies, and limitations.",
                align="C",
            )
            continue

        W = 170  # usable page width (210 - 20 left - 20 right margins)

        # Section title
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(20, 50, 100)
        pdf.set_x(20)
        pdf.multi_cell(W, 8, _safe(section["title"]), align="L")
        pdf.set_draw_color(20, 50, 100)
        pdf.set_line_width(0.5)
        y = pdf.get_y()
        pdf.line(20, y, 190, y)
        pdf.ln(5)

        # Body text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)

        body = _safe(section["body"])
        paragraphs = body.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Sub-headers: paragraph starts with "N.M " (e.g. "3.1 Target...")
            if para[0].isdigit() and "." in para[:5]:
                first_newline = para.find("\n")
                if first_newline != -1:
                    sub_header = para[:first_newline].strip()
                    sub_body = para[first_newline + 1:].strip()
                else:
                    sub_header = para
                    sub_body = ""

                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(40, 80, 140)
                pdf.set_x(20)
                pdf.multi_cell(W, 6, sub_header, align="L")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(40, 40, 40)
                if sub_body:
                    pdf.set_x(20)
                    pdf.multi_cell(W, 6, sub_body, align="L")
            else:
                pdf.set_x(20)
                pdf.multi_cell(W, 6, para, align="L")
            pdf.ln(3)

    output_path = Path("data/sample_ai_healthcare.pdf")
    pdf.output(str(output_path))
    print(f"Sample PDF generated: {output_path}")
    print(f"Pages: {pdf.page_no()}")
    return str(output_path)


if __name__ == "__main__":
    generate()
