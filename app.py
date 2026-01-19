import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
import numpy as np
import plotly.graph_objects as go
import re

# =========================================================
# 1. PAGE CONFIGURATION & DESIGN SYSTEM
# =========================================================

st.set_page_config(
    page_title="ViroAI Research Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Updated styling with new color scheme ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');

/* GLOBAL */
html, body, .stApp {
    background-color: #E6E6FA; /* Light Lavender Accent */
    font-family: 'Source Sans Pro', sans-serif;
    color: #4B0082; /* Indigo */
}

/* HEADERS */
h1, h2, h3 {
    color: #5D3A9B; /* Deep Purple */
    font-weight: 700;
}

/* HEADER BOX */
.app-header {
    background: linear-gradient(135deg, #6B7B8C 0%, #4682B4 100%); /* Blue-Gray to Steel Blue */
    padding: 1.6rem;
    border-radius: 6px;
    margin-bottom: 2rem;
    border: 1px solid #E6E6FA; /* Light Lavender */
}

/* BETTER TABLE CONTRAST */
.stDataFrame, .stTable {
    background-color: #FFF8E7 !important; /* Soft Cream */
}
div[data-testid="stDataFrame"] table {
    background-color: #FFF8E7 !important;
    color: #4B0082 !important; /* Indigo */
}
div[data-testid="stDataFrame"] th {
    background-color: #E6E6FA !important; /* Light Lavender */
    color: #5D3A9B !important; /* Deep Purple */
    font-weight: 600;
}
div[data-testid="stDataFrame"] td {
    color: #4B0082 !important; /* Indigo */
}
div[data-testid="stDataFrame"] tr:nth-child(even) {
    background-color: #F0F0FA !important; /* Slight variation of Light Lavender */
}

/* METRIC STYLING */
[data-testid="stMetricValue"] {
    color: #5D3A9B !important; /* Deep Purple */
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #4682B4 !important; /* Steel Blue */
}

/* BUTTONS */
.stButton > button {
    background-color: #5D3A9B; /* Deep Purple */
    color: #FFF8E7; /* Soft Cream */
    border: none;
    border-radius: 6px;
    padding: 0.8rem 1.6rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #4B0082; /* Indigo */
}

/* TABS */
.stTabs [data-baseweb="tab-list"] { gap: 2.2rem; }
.stTabs [data-baseweb="tab"] {
    height: 3.2rem;
    font-weight: 600;
    font-size: 1.1rem;
    color: #6B7B8C; /* Blue-Gray */
}
.stTabs [aria-selected="true"] {
    color: #5D3A9B !important; /* Deep Purple */
    border-bottom-color: #5D3A9B !important;
}

/* DOC BOXES */
.doc-box {
    background: #FFF8E7; /* Soft Cream */
    border-left: 5px solid #5D3A9B; /* Deep Purple */
    padding: 1.4rem;
    margin: 1rem 0;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.doc-title {
    color: #5D3A9B; /* Deep Purple */
    font-weight: 700;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. MODEL ARCHITECTURE & LOGIC
# =========================================================

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(320, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
    def forward(self, v, h):
        return self.encoder(v), self.encoder(h)

@st.cache_resource
def load_system():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    siamese = SiameseNet()
    try:
        siamese.load_state_dict(torch.load('viroai_siamese.pth', map_location=torch.device('cpu')))
    except:
        pass
    siamese.eval()
    return tokenizer, esm, siamese

def validate_sequence(seq):
    clean_seq = re.sub(r'\s+', '', seq).upper()
    if not clean_seq:
        return False, ""
    aa_alphabet = set("ACDEFGHIKLMNPQRSTVWYX")
    if all(c in aa_alphabet for c in clean_seq):
        return True, clean_seq
    return False, ""

def get_affinity_score(dist, margin=1.8, k=4):
    return 1 / (1 + np.exp(k * (dist - margin))) * 100

# =========================================================
# 3. DATA ASSETS
# =========================================================

HUMAN_LIBRARY = {
    "ACE2 (Respiratory)": "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQMH",
    "DPP4 (MERS-CoV)": "MKTPWKVLLGLLGAAALVTIITVPVVLLNKGTDDATADSRKTYTLTDYLKNTYRLKLYSLRWISDHEYLYKQENNILVFNAEYGNSSVFLENSTFDEFGHSINDYSISPDGQFILLEYNYVKQWRHSYTASYDIYDLNKRQLITEERIPNNTQWVTWSPVGHKLAYVWNNDIYVKIEPNLPSYRITWTGKEDIIYNGITDWVYEEEVFSAYSALWWSPNGTFLAYAQFNDTEVPLIEYSFYSDESLQYPKTVRVPYPKAGAVNPTVKFFVVNTDSLSSVTNATSIQITAPASMLIGDHYLCDVTWATQERISLQWLRRIQNYSVMDICDYDESSGRWNCLVARQHIEMSTTGWVGRFRPSEPHFTLDGNSFYKIISNEEGYRHICYFQIDKKDCTFITKGTWEVIGIEALTSDYLYYISNEYKGMPGGRNLYKIQLSDYTKVTCLSCELNPERCQYYSVSFSKEAKYYQLRCSGPGLPLYTLHSSVNDKGLRVLEDNSALDKMLQNVQMPSKKLDFIILNETKFWYQMILPPHFDKSKKYPLLLDVYAGPCSQKADTVFRLNWATYLASTENIIVASFDGRGSGYQGDKIMHAINRRLGTFEVEDQIEAARQFSKMGFVDNKRIAIWGWSYGGYVTSMVLGSGSGVFKCGIAVAPVSRWEYYDSVYTERYMGLPTPEDNLDHYRNSTVMSRAENFKQVEYLLIHGTADDNVHFQQSAQISKALVDVGVDFQAMWYTDEDHGIASSTAHQHIYTHMSHFIKQCFSLP",
    "ANPEP (HCoV-229E)": "MAKGFYISKSLGILGILLGVAAVCTIIALSVVYCQEKRQKNSKDSVAYPVTEERAALPLGSSGLVAPARSSLPSSSNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPSSGNLPssGNLPSSGNLPSSGNLPSSGNLPssGNLP",
    "CD4 (Immune)": "MNRGVPFRHLLLVLQLALLPAATQGKKVVLGKKGDTVELTCTASQKKSIQFHWKNSNQIKILGNQGSFLTKGPSKLNDRADSRRSLWDQGNFPLIIKNLKIEDSDTYICEVEDQKEEVQLLVFGLTANSDTHLLQGQSLTLTLESPPGSSPSVQCRSPRGKNIQGGKTLSVSQLELQDSGTWTCTVLQNQKKVEFKIDIVVLAFQKASSIVYKKEGEQVEFSFPLAFTVEKLTGSGELWWQAERASSSKSWITFDLKNKEVSVKRVTQDPKLQMGKKLPLHLTLPQALPQYAGSGNLTLALEAKTGKLHQEVNLVVMRATQLQKNLTCEVWGPTSPKLMLSLKLENKEAKVSKREKAVWVLNPEAGMWQCLLSDSGQVLLESNIKVLPTWSTPVQPM",
    "NPC1 (Ebola)": "MTARGLALGLVLLLLCPQAIAEPVVWVNPEGVVIGSSKILNCAVDSGTVSTVNWKDGPLVKTLDNRKAFKPGYPLIIDKikIDDSDTYICEVEDQKEEVQLLVFG",
    "CCR5 (HIV Co-receptor)": "MDYQVSSPIYDINYYTSEPCQKINVKQIAARLLPPLYSLVFILLFGNTLVMVLILINCKRLKSMTDIYLLNLAISDLFFLLTVPFWAHYAAAQWDFGNTMCQLLTGLYFIGFFSGIFFIILLTIDRYLAVVHAVFALKARTVTFGVVTSVITWVVAVFASLPGIIFTRSQKEGLHYTCSSHFPYSQYQFWKNFQTLKIVILGLVLPLLVMVICYSGILKTLLRCRNEKKRHRAVRLIFTIMIVYFLFWAPYNIVLLLNTFQEFFGLNNCSSSNRLDQAMQVTETLGMTHCCINPIIYAFVGEKFRNYLLVFFQKHIAKRFCKCCSIFQQEAPERASSVYTRSTGEQEISVGL",
    "CHRNA1 (Rabies)": "MEPWPLLLLFSLCSAGLVLGSEHETRLVAKLFKDYSSVVRPVEDHRQVVEVTVGLQLIQLINVDEVNQIVTTNVRLKQQWVDYNLKWNPDDYGGVKKIHIPSEKIWRPDLVLYNNADGDFAIVKFTKVLLDYTGKIMWTPPAIFKSYCEIIVTHFPFDQQNCTMKLGIWTYDGSVVAINPESDQPDLSNFMESGEWVIKESRGWKHSVTYSCCPDTPYLDITYHFVMQRLPLYFIVNVIIPCLLFSFLTGLVFYLPTDSGEKMTLSISVLLSLTVFLLVIVELIPSTSSAVPLIGKYMLFTMVFVIASIIITVIVINTHHRSPSTHVMPNWVRKVFIDTIPNIMFFSTMKRPSREKQDKKIFTEDIDISDISGKPGPPPMGFHSPLIKHPEVKSAIEGVKYIAETMKSDQESNNAAAEWKYVAMVMDHILLGVFMLVCIIGTLAVFAGRLIELHQQG",
    "PVR (Poliovirus)": "MARAMAAAWPLLLVALLVLSWPPPGTGDVVVQAPTQVPGFLGDSVTLPCYLQVPNMEVTHVSQLTWARHGESGSMAVFHQTQGPSYSESKRLEFVAARLGAELRNASLRMFGLRVEDEGNYTCLFVTFPQGSRSVDIWLRVLAKPQNTAEVQKVQLTGEPVPMARCVSTGGRPPAQITWHSDLGGMPNTSQVPGFLSGTVTVTSLWILVPSSQVDGKNVTCKVEHESFEKPQLLTVNLTVYYPPEVSISGYDNNWYLGQNEATLTCDARSNPEPTGYNWSTTMGPLPPFAVAQGAQLLIRPVDKPINTTLICNVTNALGARQAELTVQVKEGPPSEHSGMSRNA",
    "CD81 (Hepatitis C)": "MGVEGCTKCIKYLLFVFNFVFWLAGGVILGVALWLRHDPQTTNLLYLELGDKPAPNTFYVGIYILIAVGAVMMFVGFLGCYGAIQESQCLLGTFFTCLVILF ACEVAAAIWGFVNYDQAKEVLKKFMDDTLKYCLGNTLDRMQADFKKCCGVNGVDW",
}

# =========================================================
# 4. LOAD MODEL
# =========================================================

tokenizer, esm, siamese = load_system()

# ── Header with logo and text side by side ──────────────────────────────────
col_logo, col_text = st.columns([1, 4])
with col_logo:
    try:
        st.image("logo.png", width=160)
    except:
        st.markdown("<h1 style='color:#5D3A9B; font-size:2.2rem; margin:0;'>Viro-AI</h1>", unsafe_allow_html=True)
with col_text:
    st.markdown("<h1 style='margin:0; color:#5D3A9B; font-size:2.2rem;'>ViroAI Research Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0; color:#4682B4; font-size:1.1rem;'>Precision Metric Learning for Viral-Host Structural Interaction Prediction</p>", unsafe_allow_html=True)

# ── Documentation Expander ──────────────────────────────────────────────────
with st.expander("PLATFORM DOCUMENTATION: Workflow & Logic", expanded=True):
    st.markdown("""
    **Core Principle**  
    This tool uses **deep metric learning** (not sequence alignment).  
    ESM-2 → protein embeddings → Siamese network → Euclidean distance → binding probability.

    **Workflow**  
    1. Input amino acid sequences (standard IUPAC one-letter codes)  
    2. Tokenize & embed with ESM-2  
    3. Project into shared space with Siamese net  
    4. Compute distance → convert to affinity probability (0–100%)

    **Modules**  
    • **Bilateral Analysis** → test one virus vs one receptor  
    • **Multi-Target Discovery** → screen one virus vs library of human receptors
    """)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Bilateral Analysis", "Multi-Target Discovery"])

# =========================================================
# TAB 1: BILATERAL ANALYSIS
# =========================================================
with tab1:
    st.markdown("""
    <div class="doc-box">
        <span class="doc-title">BILATERAL ANALYSIS</span>
        Test a specific virus–receptor hypothesis.<br><br>
        1. Paste viral surface protein sequence (left)<br>
        2. Paste human receptor sequence (right)<br>
        3. Click "Run Analysis"
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Viral Sequence**")
        v_raw = st.text_area("Viral", placeholder="Paste viral AA sequence...", height=140, label_visibility="collapsed", key="bilateral_v")
    with c2:
        st.markdown("**Human Receptor Sequence**")
        h_raw = st.text_area("Human", placeholder="Paste human AA sequence...", height=140, label_visibility="collapsed", key="bilateral_h")

    if st.button("Run Bilateral Analysis", type="primary", use_container_width=True):
        is_v, v_seq = validate_sequence(v_raw)
        is_h, h_seq = validate_sequence(h_raw)

        if is_v and is_h:
            with st.spinner("Encoding sequences • Computing embedding distance..."):
                v_emb = esm(**tokenizer(v_seq[:1022], return_tensors="pt")).last_hidden_state.mean(dim=1)
                h_emb = esm(**tokenizer(h_seq[:1022], return_tensors="pt")).last_hidden_state.mean(dim=1)

                with torch.no_grad():
                    v_vec, h_vec = siamese(v_emb, h_emb)
                    dist = torch.pairwise_distance(v_vec, h_vec).item()

                score = get_affinity_score(dist)

            # ── Metrics ─────────────────────────────────────────────────────
            st.markdown("### Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Binding Probability", f"{score:.1f}%")
            col2.metric("Distance", f"{dist:.3f}")
            col3.metric("Risk Level", "CRITICAL" if score > 75 else "MODERATE" if score > 40 else "LOW")
            col4.metric("Affinity Strength", "Strong" if score > 75 else "Possible" if score > 40 else "Unlikely")

            # ── Risk Gauge ──────────────────────────────────────────────────
            gauge_color = "#d32f2f" if score > 75 else "#f57c00" if score > 40 else "#4caf50"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Binding Risk (%)"},
                number={'font': {'size': 38}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#E6E6FA"},
                        {'range': [40, 75], 'color': "#FFF8E7"},
                        {'range': [75, 100], 'color': "#F0E6FA"}
                    ],
                    'threshold': {
                        'line': {'color': "#b71c1c", 'width': 4},
                        'thickness': 0.8,
                        'value': 75
                    }
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Improved Association Visualization ───────────────────────────
            fig_bridge = go.Figure()

            fig_bridge.add_trace(go.Scatter(
                x=[-0.7, 0.7], y=[0, 0],
                mode='markers+text',
                marker=dict(size=64, color=['#5D3A9B', '#4682B4'], line=dict(width=2.5, color='#FFF8E7')),
                text=['VIRAL', 'HUMAN'],
                textposition='bottom center',
                textfont=dict(size=15, color='#FFF8E7')
            ))

            fig_bridge.add_trace(go.Scatter(
                x=[-0.7, 0, 0.7],
                y=[0, 0.45 * (1 - dist/3.5), 0],
                mode='lines',
                line=dict(color='#6B7B8C', width=max(4, score/7), shape='spline')
            ))

            fig_bridge.add_trace(go.Scatter(
                x=[0], y=[0.26],
                mode='text',
                text=[f"Distance = {dist:.2f}"],
                textfont=dict(size=16, color="#4B0082")
            ))

            fig_bridge.update_layout(
                height=360,
                showlegend=False,
                xaxis=dict(visible=False, range=[-1.1, 1.1]),
                yaxis=dict(visible=False, range=[-0.5, 0.7]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(fig_bridge, use_container_width=True)

            # ── Quick biochemical profile (rule-of-thumb) ───────────────────
            def approx_hydrophobicity(seq):
                hydro = sum(1 for c in seq if c in "AILFWVPM") / max(1, len(seq)) * 100
                return round(hydro, 1)

            def approx_net_charge(seq):
                pos = seq.count("K") + seq.count("R")
                neg = seq.count("D") + seq.count("E")
                return pos - neg

            with st.expander("Quick Physicochemical Profile"):
                ca, cb = st.columns(2)
                ca.metric("Viral Hydrophobicity", f"{approx_hydrophobicity(v_seq)}%")
                cb.metric("Human Hydrophobicity", f"{approx_hydrophobicity(h_seq)}%")
                ca.metric("Viral Net Charge", approx_net_charge(v_seq))
                cb.metric("Human Net Charge", approx_net_charge(h_seq))

            # Interpretation
            if score > 75:
                interp = "Very strong predicted interaction – high structural compatibility."
            elif score > 40:
                interp = "Moderate compatibility – interaction possible under specific conditions."
            else:
                interp = "Low predicted binding likelihood – large structural divergence."

            st.info(f"**Interpretation**: {interp}\n\nTechnical: Probability from inverse sigmoid of distance (d = {dist:.3f})")

        else:
            st.error("Invalid sequence(s). Use only standard amino acids (A–Y).")

# =========================================================
# TAB 2: MULTI-TARGET DISCOVERY
# =========================================================
with tab2:
    st.markdown("""
    <div class="doc-box">
        <span class="doc-title">MULTI-TARGET DISCOVERY</span>
        Screen one viral protein against a library of known human receptors.<br><br>
        1. Paste viral sequence below<br>
        2. Click "Scan Library"<br>
        3. Review ranked results
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Viral Sequence**")
    v_raw_multi = st.text_area("Viral protein sequence", height=160, label_visibility="collapsed", key="multi_v")

    if st.button("Scan Internal Library", type="primary", use_container_width=True):
        is_v, v_seq = validate_sequence(v_raw_multi)
        if is_v:
            with st.spinner("Encoding virus • Comparing against receptor library..."):
                v_emb = esm(**tokenizer(v_seq[:1022], return_tensors="pt")).last_hidden_state.mean(dim=1)
                with torch.no_grad():
                    v_vec, _ = siamese(v_emb, v_emb)

                results = []
                for name, seq in HUMAN_LIBRARY.items():
                    h_emb = esm(**tokenizer(seq[:1022], return_tensors="pt")).last_hidden_state.mean(dim=1)
                    with torch.no_grad():
                        _, h_vec = siamese(v_emb, h_emb)
                        d = torch.pairwise_distance(v_vec, h_vec).item()
                    results.append((name, d))

            # ── Hub-and-spoke graph ─────────────────────────────────────────
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=54, color='#5D3A9B'),
                text=["VIRUS"],
                textposition="middle center",
                textfont=dict(color="#FFF8E7", size=14)
            ))

            for i, (name, d) in enumerate(results):
                angle = 2 * np.pi * i / max(1, len(results))
                r = min(1.3, max(0.35, d * 0.65))
                x, y = r * np.cos(angle), r * np.sin(angle)
                prob = get_affinity_score(d)
                w = max(1.8, prob / 11)

                fig.add_trace(go.Scatter(
                    x=[0, x], y=[0, y],
                    mode='lines',
                    line=dict(color='#4682B4', width=w),
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=34, color='#6B7B8C'),
                    text=[name.split(" (")[0]],
                    textposition="middle center",
                    textfont=dict(color="#FFF8E7", size=11),
                    hovertext=f"{name}<br>Dist: {d:.3f}<br>Aff: {prob:.1f}%",
                    hoverinfo="text"
                ))

            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Affinity Table ──────────────────────────────────────────────
            st.markdown("### Affinity Report (sorted by predicted strength)")
            table_data = [
                {
                    "Receptor": r[0],
                    "Distance": round(r[1], 3),
                    "Affinity (%)": round(get_affinity_score(r[1]), 1),
                    "Risk": "HIGH" if r[1] < 1.5 else "MEDIUM" if r[1] < 2.2 else "LOW"
                }
                for r in sorted(results, key=lambda x: x[1])
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            # Interpretation
            if results:
                top_name, top_dist = sorted(results, key=lambda x: x[1])[0]
                top_prob = get_affinity_score(top_dist)
                st.info(f"**Top predicted receptor**: **{top_name}** (distance = {top_dist:.3f} • affinity ≈ {top_prob:.1f}%)")
                st.caption("Focus experimental validation on the top 2–4 ranked targets.")

        else:
            st.error("Invalid viral sequence. Please use only standard amino acids.")

st.caption("© 2026 ViroAI Research Platform • Powered by ESM-2 + Metric Learning")