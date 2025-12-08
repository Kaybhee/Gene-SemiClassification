"""
RAG (Retrieval-Augmented Generation) module for gene signature classification.
Provides biomolecular context and scientific insights by querying local knowledge bases.
Includes gene annotations, pathway information, disease associations, and therapeutic targets.
"""

import logging
import json
import re
from typing import Dict, List, Optional

logger = logging.getLogger("rag")

GENE_KNOWLEDGE = {
    "MYC": {
        "symbol": "MYC",
        "full_name": "MYC proto-oncogene",
        "functions": ["transcription factor", "cell proliferation", "apoptosis regulation"],
        "pathways": ["KEGG:hsa05200 (Pathways in cancer)", "KEGG:hsa05222 (Small cell lung cancer)"],
        "diseases": ["Burkitt lymphoma", "colorectal cancer", "breast cancer", "multiple myeloma"],
        "regulation": "Often upregulated in cancer; target of multiple signaling pathways",
        "therapeutic_targets": ["BET inhibitors", "CDK inhibitors", "AURKA inhibitors"]
    },
    "TP53": {
        "symbol": "TP53",
        "full_name": "Tumor suppressor p53",
        "functions": ["DNA damage response", "cell cycle arrest", "apoptosis", "genomic stability"],
        "pathways": ["KEGG:hsa05200 (Pathways in cancer)", "KEGG:hsa04115 (p53 signaling pathway)"],
        "diseases": ["Li-Fraumeni syndrome", "various cancers (p53 mutation)"],
        "regulation": "Master tumor suppressor; activated by DNA damage stress",
        "therapeutic_targets": ["MDM2 inhibitors", "ATM kinase inhibitors", "p53 restoration therapy"]
    },
    "EGFR": {
        "symbol": "EGFR",
        "full_name": "Epidermal growth factor receptor",
        "functions": ["tyrosine kinase receptor", "cell growth", "proliferation", "differentiation"],
        "pathways": ["KEGG:hsa04012 (ErbB signaling pathway)", "KEGG:hsa05200 (Pathways in cancer)"],
        "diseases": ["non-small cell lung cancer (EGFR mutations)", "colorectal cancer", "head and neck cancer"],
        "regulation": "Growth factor-dependent; commonly mutated/amplified in cancer",
        "therapeutic_targets": ["EGFR tyrosine kinase inhibitors (erlotinib, gefitinib)", "monoclonal antibodies (cetuximab)"]
    },
    "BRCA1": {
        "symbol": "BRCA1",
        "full_name": "Breast cancer susceptibility protein 1",
        "functions": ["DNA repair", "transcription regulation", "cell cycle control"],
        "pathways": ["KEGG:hsa03440 (Homologous recombination)"],
        "diseases": ["hereditary breast and ovarian cancer (BRCA1 mutations)", "pancreatic cancer"],
        "regulation": "Induced after DNA damage; essential for homologous recombination repair",
        "therapeutic_targets": ["PARP inhibitors (olaparib, rucaparib)", "platinum agents"]
    },
    "JUN": {
        "symbol": "JUN",
        "full_name": "Jun proto-oncogene",
        "functions": ["transcription factor (AP-1 complex)", "stress response", "inflammation"],
        "pathways": ["KEGG:hsa04010 (MAPK signaling pathway)"],
        "diseases": ["inflammation-associated cancers", "stress response disorders"],
        "regulation": "Induced by stress, growth factors, cytokines; part of immediate early genes",
        "therapeutic_targets": ["MEK inhibitors", "c-Jun N-terminal kinase (JNK) inhibitors"]
    },
    "FOS": {
        "symbol": "FOS",
        "full_name": "Fos proto-oncogene",
        "functions": ["transcription factor (AP-1 complex)", "cell proliferation", "differentiation"],
        "pathways": ["KEGG:hsa04010 (MAPK signaling pathway)"],
        "diseases": ["osteosarcoma (FOS fusion)", "inflammation-related disorders"],
        "regulation": "Immediate early gene; induced by growth factors and stress signals",
        "therapeutic_targets": ["MAPK pathway inhibitors"]
    },
    "STAT1": {
        "symbol": "STAT1",
        "full_name": "Signal transducer and activator of transcription 1",
        "functions": ["transcription factor", "interferon signaling", "antiviral response", "innate immunity"],
        "pathways": ["KEGG:hsa04630 (Jak-STAT signaling pathway)"],
        "diseases": ["increased resistance to viral infection (STAT1 gain-of-function)", "immunodeficiency (STAT1 loss-of-function)"],
        "regulation": "Activated by Type I and Type II interferons; critical for antiviral response",
        "therapeutic_targets": ["JAK inhibitors (baricitinib)", "STAT1 selective inhibitors"]
    },
    "IFIT1": {
        "symbol": "IFIT1",
        "full_name": "Interferon-induced protein with tetratricopeptide repeats 1",
        "functions": ["antiviral response", "protein binding", "translation control"],
        "pathways": ["Interferon signaling", "antiviral defense"],
        "diseases": ["viral infection response (upregulated)", "chronic inflammation"],
        "regulation": "Strongly induced by Type I interferon and viral infection; ISG member",
        "therapeutic_targets": ["Interferon therapy", "antiviral agents"]
    },
    "IFIT3": {
        "symbol": "IFIT3",
        "full_name": "Interferon-induced protein with tetratricopeptide repeats 3",
        "functions": ["antiviral response", "protein interactions"],
        "pathways": ["Interferon signaling"],
        "diseases": ["viral response marker (upregulated in infections)"],
        "regulation": "ISG; upregulated by interferon and viral PAMPs",
        "therapeutic_targets": ["Interferon-based therapy"]
    },
    "CXCL10": {
        "symbol": "CXCL10",
        "full_name": "C-X-C motif chemokine ligand 10 (IP-10)",
        "functions": ["chemokine", "immune cell recruitment", "antiviral immunity"],
        "pathways": ["Chemokine signaling", "interferon response"],
        "diseases": ["chronic inflammation", "viral infection", "autoimmune diseases"],
        "regulation": "Induced by interferon-gamma and TNF-alpha; recruits T cells and monocytes",
        "therapeutic_targets": ["CXCR3 antagonists", "chemokine modulators"]
    },
    "CDH1": {
        "symbol": "CDH1",
        "full_name": "Cadherin-1 (E-cadherin)",
        "functions": ["cell adhesion", "epithelial integrity", "cell-cell communication"],
        "pathways": ["Adherens junctions", "Wnt/beta-catenin signaling"],
        "diseases": ["hereditary diffuse gastric cancer (CDH1 mutations)", "epithelial-mesenchymal transition (EMT)"],
        "regulation": "Downregulated during EMT and metastatic progression",
        "therapeutic_targets": ["Wnt pathway inhibitors", "cell adhesion modulators"]
    },
    "VIM": {
        "symbol": "VIM",
        "full_name": "Vimentin",
        "functions": ["cytoskeleton", "cell migration", "EMT marker"],
        "pathways": ["Cytoskeletal organization", "EMT signaling"],
        "diseases": ["metastatic cancer (high vimentin = EMT)", "fibrotic diseases"],
        "regulation": "Upregulated during EMT; marker of mesenchymal phenotype",
        "therapeutic_targets": ["EMT inhibitors", "TGF-beta signaling inhibitors"]
    },
    "KRAS": {
        "symbol": "KRAS",
        "full_name": "KRAS proto-oncogene GTPase",
        "functions": ["small GTPase", "signal transduction", "cell proliferation"],
        "pathways": ["KEGG:hsa04010 (MAPK signaling)", "KEGG:hsa05200 (Pathways in cancer)"],
        "diseases": ["pancreatic cancer (KRAS mutations ~90%)", "colorectal cancer", "lung cancer"],
        "regulation": "Frequently mutated in cancer; constitutive activation drives proliferation",
        "therapeutic_targets": ["KRAS G12C inhibitors (sotorasib, adagrasib)", "MEK/ERK inhibitors"]
    }
}

SIGNATURE_PROFILES = {
    "ctrl": {
        "name": "Control / Normal",
        "description": "Baseline, untreated gene expression state",
        "characteristics": [
            "low proliferation markers (low MYC, EGFR)",
            "normal cell adhesion (high CDH1)",
            "epithelial phenotype (low VIM)",
            "no or low stress response"
        ],
        "typical_genes": ["CDH1"],
        "biological_context": "Homeostatic gene expression in normal/baseline conditions",
        "implications": "Sample exhibits characteristics of an untreated or control state."
    },
    "pert": {
        "name": "Perturbation / Treatment",
        "description": "Response to stimulus, treatment, or pathological condition",
        "characteristics": [
            "cell stress response (JUN, FOS upregulation)",
            "altered proliferation or differentiation",
            "potential immune activation (STAT1, interferons)",
            "metabolic remodeling"
        ],
        "typical_genes": ["JUN", "FOS", "STAT1"],
        "biological_context": "Dynamic response to external stimulus or pathological trigger",
        "implications": "Sample shows active response to treatment, infection, or stress; warrants investigation into underlying stimulus."
    }
}


def extract_genes_from_text(text: str) -> List[str]:
    """Extract likely gene symbols from text (uppercase 2-10 letter words)."""
    words = re.findall(r'\b[A-Z]{2,10}\b', text)
    return list(set(words))


def retrieve_gene_info(gene_symbol: str) -> Optional[Dict]:
    """Retrieve curated knowledge about a gene from local KB."""
    return GENE_KNOWLEDGE.get(gene_symbol.upper())


def enrich_with_gene_annotations(genes: List[str], max_genes: int = 5) -> List[Dict]:
    """Retrieve annotations for discovered genes."""
    enriched = []
    for gene in genes[:max_genes]:
        info = retrieve_gene_info(gene)
        if info:
            enriched.append({
                "symbol": gene,
                "full_name": info.get("full_name"),
                "functions": info.get("functions"),
                "diseases": info.get("diseases"),
                "pathways": info.get("pathways"),
                "therapeutic_targets": info.get("therapeutic_targets")
            })
    return enriched


def infer_biological_contexts(genes: List[str]) -> List[str]:
    """Infer likely disease/biological context from gene list."""
    contexts = []
    gene_set = set([g.upper() for g in genes])
    
    # Check for cancer markers
    cancer_markers = {"MYC", "TP53", "EGFR", "KRAS", "BRCA1"}
    if gene_set & cancer_markers:
        contexts.append("Cancer-related pathways (oncogenic transformation potential)")
    
    # Check for antiviral/interferon response
    antiviral_markers = {"STAT1", "IFIT1", "IFIT3", "CXCL10"}
    if gene_set & antiviral_markers:
        contexts.append("Antiviral/Interferon response (active immune engagement)")
    
    # Check for inflammatory response
    inflammatory_markers = {"JUN", "FOS", "CXCL10"}
    if gene_set & inflammatory_markers:
        contexts.append("Inflammatory response (stress/danger signaling)")
    
    # Check for EMT/metastasis
    emt_markers = {"VIM", "CDH1"}
    if gene_set & emt_markers:
        contexts.append("Epithelial-mesenchymal transition (EMT) pathway")
    
    return contexts


def generate_insights(signature: str, genes: List[str] = None) -> Dict:
    """Generate comprehensive RAG-augmented insights for a predicted signature."""
    genes = genes or []
    profile = SIGNATURE_PROFILES.get(signature.lower(), {"name": signature, "description": "Unknown"})
    
    insights = {
        "signature": signature,
        "profile_name": profile.get("name"),
        "profile_description": profile.get("description"),
        "profile_characteristics": profile.get("characteristics", []),
        "biological_context": profile.get("biological_context"),
        "implications": profile.get("implications"),
        "gene_annotations": enrich_with_gene_annotations(genes) if genes else [],
        "inferred_contexts": infer_biological_contexts(genes) if genes else [],
    }
    
    # Build comprehensive narrative summary
    summary_parts = [
        f"**{profile.get('name')}**: {profile.get('description')}"
    ]
    
    if insights["inferred_contexts"]:
        summary_parts.append(f"Inferred biological contexts: {'; '.join(insights['inferred_contexts'])}")
    
    if insights["gene_annotations"]:
        genes_str = ", ".join([a["symbol"] for a in insights["gene_annotations"]])
        summary_parts.append(f"Key genes identified: {genes_str}")
        
        # Extract therapeutic targets
        therapeutic_targets = set()
        for ann in insights["gene_annotations"]:
            if "therapeutic_targets" in ann:
                therapeutic_targets.update(ann["therapeutic_targets"])
        
        if therapeutic_targets:
            targets_str = "; ".join(list(therapeutic_targets)[:3])
            summary_parts.append(f"Potential therapeutic targets: {targets_str}")
    
    summary_parts.append(profile.get("implications", ""))
    
    insights["summary"] = " | ".join([p for p in summary_parts if p])
    
    return insights


def retrieve_insights(signature: str, genes: List[str] = None) -> str:
    """
    High-level function to retrieve RAG-augmented insights for a classification result.
    Returns JSON-formatted enriched insights with gene annotations, pathways, and diseases.
    """
    full_insights = generate_insights(signature, genes)
    return json.dumps(full_insights, indent=2)
