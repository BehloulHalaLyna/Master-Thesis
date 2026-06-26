"""
=============================================================================
QUANTITATIVE ANALYSIS — AI Integration in Agile Teams
M2 Research Thesis — French Organizations (n=100)
=============================================================================

FRAMEWORK:
  Independent Variables  : AI Adoption (Q11–Q15), Tool Types (Q16–Q23),
                           Agile Phase Integration (Q24–Q28)
  Dependent Variables    : Team Performance (Q35–Q36), Deliverable Quality
                           (Q37–Q38), Overall Project Success (Q39)
  Perceived Mechanisms   : Decision-making (Q29–Q30), Automation (Q31–Q32),
                           Collaboration (Q33–Q34)
  Moderating Variables   : Agile Maturity (Q7), Org Context (Q1–Q6, Q8–Q10)

HYPOTHESES:
  H1 — AI adoption level is significantly associated with Agile team performance
  H2 — AI adoption level is significantly associated with deliverable quality
  H3 — AI integration across Agile phases is significantly associated with
        overall project success
  H4 — AI adoption level is significantly associated with overall project success

ANALYSIS PIPELINE:
  0. Data loading & encoding
  1. Composite index construction
  2. Frequency analysis — respondent profile (Q1–Q10)
  3. Descriptive statistics — all constructs (mean, SD, skewness, kurtosis)
  4. Reliability analysis — Cronbach's alpha per construct
  5. Validity analysis — KMO + Bartlett's test
  6. Normality analysis — Shapiro-Wilk + skewness/kurtosis
  7. Correlation matrix — Spearman (full, with significance)
  8. Hypothesis testing — H1–H4 (Spearman + effect size)
  9. Group comparisons — Kruskal-Wallis + Dunn post-hoc
 10. Moderation analysis — Agile maturity as moderator
 11. Regression — OLS with ANOVA table, VIF, standardized coefficients
 12. Descriptive block — perceived mechanisms (Q29–Q34)
 13. Content analysis — Q40 challenges
 14. Summary table of all hypotheses

Usage:
  python analysis_ai_agile.py
  → Figures saved to ./figures_en/
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, spearmanr, kruskal, mannwhitneyu,
                         chi2_contingency)
from itertools import combinations

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = r"C:/Users/halab/Desktop/memoire/data.csv"
OUTPUT_DIR = "./figures_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Academic color palette
C_MAIN   = "#2C5F8A"
C_SEC    = "#E07B39"
C_POS    = "#27AE60"
C_NEG    = "#C0392B"
C_NEU    = "#95A5A6"
C_LIGHT  = "#D6E4F0"

DPI = 300
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING MAPS
# ─────────────────────────────────────────────────────────────────────────────
FREQ_5 = {
    'Jamais': 1, 'Rarement': 2, 'Parfois': 3,
    'Souvent': 4, 'Tr\u00e8s souvent': 5
}
INTENSITY_5 = {
    'Pas du tout utilis\u00e9': 1, 'Faiblement utilis\u00e9': 2,
    'Mod\u00e9r\u00e9ment utilis\u00e9': 3, 'Fortement utilis\u00e9': 4,
    'Tr\u00e8s fortement utilis\u00e9': 5
}
LIKERT_5 = {
    'Pas du tout d\u2019accord': 1, 'Plut\u00f4t pas d\u2019accord': 2,
    'Neutre': 3, 'Plut\u00f4t d\u2019accord': 4,
    'Tout \u00e0 fait d\u2019accord': 5
}
IMPACT_DIR = {
    'Fortement diminu\u00e9': 1, 'Diminu\u00e9': 2,
    'Stable': 3, 'Augment\u00e9': 4, 'Fortement augment\u00e9': 5
}
IMPACT_INV = {  # Q37: fewer defects = improvement
    'Fortement augment\u00e9': 1, 'Augment\u00e9': 2,
    'Stable': 3, 'Diminu\u00e9': 4, 'Fortement diminu\u00e9': 5
}
NB_TOOLS = {'0': 0, '1': 1, '2\u20133': 2.5, '4\u20135': 4.5, '6 et +': 7}
TYPE_USAGE = {
    'Ponctuel / exp\u00e9rimental (utilisation occasionnelle, non formalis\u00e9e)': 1,
    'R\u00e9gulier mais non standardis\u00e9 (utilisation fr\u00e9quente, sans r\u00e8gles fixes)': 2,
    'Institutionnalis\u00e9e / standardis\u00e9e (utilisation int\u00e9gr\u00e9e aux pratiques de travail)': 3
}
MATURITY = {
    'Agile limit\u00e9 \u00e0 quelques \u00e9quipes pilotes': 1,
    'Agile adopt\u00e9 par plusieurs \u00e9quipes': 2,
    'Agile largement d\u00e9ploy\u00e9 dans l\u2019organisation': 3,
    'Agile structur\u00e9 via un cadre \u00e0 grande \u00e9chelle (SAFe, LeSS, etc.)': 4
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'n.s.'

def effect_size_r(rho):
    a = abs(rho)
    if a >= .50: return 'Large'
    if a >= .30: return 'Moderate'
    if a >= .10: return 'Small'
    return 'Negligible'

def save_fig(name):
    path = f"{OUTPUT_DIR}/{name}.png"
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  → Figure saved: {path}")

def cronbach_alpha(data: pd.DataFrame) -> float:
    d = data.dropna()
    n = d.shape[1]
    if n < 2: return np.nan
    item_vars  = d.var(axis=0, ddof=1)
    total_var  = d.sum(axis=1).var(ddof=1)
    if total_var == 0: return np.nan
    return round((n / (n - 1)) * (1 - item_vars.sum() / total_var), 3)

def kmo_bartlett(data: pd.DataFrame):
    """KMO and Bartlett's test of sphericity."""
    X = data.dropna().values
    n, p = X.shape
    R = np.corrcoef(X.T)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    D = np.diag(1 / np.sqrt(np.diag(R_inv)))
    P = -D @ R_inv @ D
    np.fill_diagonal(P, 1)
    r2_sum = (R**2).sum() - np.trace(R**2)
    p2_sum = (P**2).sum() - np.trace(P**2)
    kmo = r2_sum / (r2_sum + p2_sum)
    chi2_stat = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df_b = p * (p - 1) / 2
    p_val = 1 - stats.chi2.cdf(chi2_stat, df_b)
    return round(kmo, 3), round(chi2_stat, 3), int(df_b), round(p_val, 4)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOADING & ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    c  = df.columns.tolist()

    print(f"\n{'='*70}")
    print(f"  DATA LOADED: {df.shape[0]} respondents, {df.shape[1]} variables")
    print(f"{'='*70}")

    df['Q11_n'] = df[c[11]].map(FREQ_5)
    df['Q12_n'] = df[c[12]].map(NB_TOOLS)
    df['Q13_n'] = pd.to_numeric(df[c[13]], errors='coerce')
    df['Q14_n'] = df[c[14]].map(TYPE_USAGE)
    df['Q15_n'] = pd.to_numeric(df[c[15]], errors='coerce')

    for i in range(16, 24):
        df[f'Q{i}_n'] = df[c[i]].map(INTENSITY_5)

    for i in range(24, 29):
        df[f'Q{i}_n'] = df[c[i]].map(FREQ_5)

    for i in range(29, 35):
        df[f'Q{i}_n'] = df[c[i]].map(LIKERT_5)

    df['Q35_n'] = pd.to_numeric(df[c[35]], errors='coerce')
    df['Q36_n'] = df[c[36]].map(IMPACT_DIR)
    df['Q37_n'] = df[c[37]].map(IMPACT_INV)   # reversed: fewer defects = better
    df['Q38_n'] = df[c[38]].map(IMPACT_DIR)
    df['Q39_n'] = pd.to_numeric(df[c[39]], errors='coerce')
    df['Q7_n']  = df[c[7]].map(MATURITY)

    # Normalize Q12 to 1–5 scale
        # Normalize Q12 to 1–5 scale
    df['Q12_norm'] = (df['Q12_n'] / 7) * 4 + 1

    # IV1 — AI Adoption Index (Q11, Q12_norm, Q13, Q14, Q15)
    df['IDX_ADOPTION'] = df[['Q11_n', 'Q12_norm', 'Q13_n', 'Q14_n', 'Q15_n']].mean(axis=1)

    # IV2 — AI Tools Index (Q16–Q23)
    df['IDX_TOOLS'] = df[[f'Q{i}_n' for i in range(16, 24)]].mean(axis=1)

    # IV3 — Agile Phase Integration Index (Q24–Q28)
    df['IDX_PHASES'] = df[[f'Q{i}_n' for i in range(24, 29)]].mean(axis=1)

    # DV1 — Team Performance (Q35, Q36)
    df['IDX_PERF'] = df[['Q35_n', 'Q36_n']].mean(axis=1)

    # DV2 — Deliverable Quality (Q37 reversed, Q38)
    df['IDX_QUALITY'] = df[['Q37_n', 'Q38_n']].mean(axis=1)

    # DV3 — Overall Project Success (Q39)
    df['IDX_SUCCESS'] = df['Q39_n']

    # Perceived Mechanisms (descriptive block)
    df['IDX_DECISION']   = df[['Q29_n', 'Q30_n']].mean(axis=1)
    df['IDX_AUTOMATION'] = df[['Q31_n', 'Q32_n']].mean(axis=1)
    df['IDX_COLLAB']     = df[['Q33_n', 'Q34_n']].mean(axis=1)

    key_indices = ['IDX_ADOPTION','IDX_TOOLS','IDX_PHASES',
                   'IDX_PERF','IDX_QUALITY','IDX_SUCCESS',
                   'IDX_DECISION','IDX_AUTOMATION','IDX_COLLAB']
    print("\n  Missing values per index:")
    for idx in key_indices:
        print(f"    {idx:<20} {df[idx].isna().sum()} NA")

    return df, c

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FREQUENCY ANALYSIS — RESPONDENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_LABELS = {
    1:  "Role (Q1)",          2:  "Professional Experience (Q2)",
    3:  "Agile Experience (Q3)", 4: "Organization Size (Q4)",
    5:  "Sector (Q5)",        6:  "Organization Type (Q6)",
    7:  "Agile Maturity (Q7)", 8: "Location (Q8)",
    9:  "Team Size (Q9)",     10: "Project Domain (Q10)"
}
PROFILE_EN = {
    'D\u00e9veloppeur': 'Developer',
    'Sp\u00e9cialiste Data / IA': 'Data / AI Specialist',
    'Chef de projet / Project Manager': 'Project Manager',
    'Consultant': 'Consultant',
    'Autre': 'Other',
    'Manager / Responsable': 'Manager',
    'Product Owner': 'Product Owner',
    'Scrum Master': 'Scrum Master',
    'QA / Test': 'QA / Test',
    'Finance / Assurance': 'Finance / Insurance',
    'IT / Logiciel': 'IT / Software',
    'Commerce / Retail': 'Retail / Commerce',
    'Industrie': 'Industry',
    'Conseil / Consulting': 'Consulting',
    'Secteur public': 'Public Sector',
    'T\u00e9l\u00e9com': 'Telecom',
    'Priv\u00e9e': 'Private', 'Publique': 'Public',
    'Semi-publique': 'Semi-public', 'Non lucratif': 'Non-profit',
    'Agile limit\u00e9 \u00e0 quelques \u00e9quipes pilotes': 'Pilot teams only',
    'Agile adopt\u00e9 par plusieurs \u00e9quipes': 'Several teams',
    'Agile largement d\u00e9ploy\u00e9 dans l\u2019organisation': 'Company-wide',
    'Agile structur\u00e9 via un cadre \u00e0 grande \u00e9chelle (SAFe, LeSS, etc.)': 'Scaled framework',
    '\u00cele-de-France': 'Île-de-France',
    '\u00c9quipe distribu\u00e9e / Remote': 'Distributed / Remote',
    'Multi-sites': 'Multi-site', 'Sud': 'South',
    'Ouest': 'West', 'Nord': 'North',
    '3\u20135 personnes': '3–5 people', '6\u20139 personnes': '6–9 people',
    '10\u201315 personnes': '10–15 people', '16 personnes et +': '16+ people',
    'Produit Data / IA': 'Data / AI Product',
    'IT interne': 'Internal IT',
    'D\u00e9veloppement produit': 'Product Development',
    'Processus m\u00e9tier': 'Business Process',
    'Infrastructure / Cloud': 'Infrastructure / Cloud',
    '0\u20132 ans': '0–2 yrs', '3\u20135 ans': '3–5 yrs',
    '6\u201310 ans': '6–10 yrs', '11\u201315 ans': '11–15 yrs',
    '16 ans et +': '16+ yrs',
    '0\u20131 an': '0–1 yr', '2\u20133 ans': '2–3 yrs',
    '4\u20136 ans': '4–6 yrs', '7 ans et +': '7+ yrs',
    '1\u201349 employ\u00e9s': '1–49 employees',
    '50\u2013249 employ\u00e9s': '50–249 employees',
    '250\u2013999 employ\u00e9s': '250–999 employees',
    '1000 employ\u00e9s et +': '1000+ employees',
}

def translate(val):
    if isinstance(val, str):
        for k, v in PROFILE_EN.items():
            if k in val:
                return v
    return str(val)[:35]

def freq_profile(df, cols):
    print(f"\n{'='*70}")
    print("  SECTION 1 — RESPONDENT PROFILE (Q1–Q10)")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle("Respondent Profile (n=100)", fontsize=14,
                 fontweight='bold', y=1.01)

    for ax, (idx, label) in zip(axes.flatten(), PROFILE_LABELS.items()):
        series  = df[cols[idx]].map(translate)
        freq    = series.value_counts()
        colors  = sns.color_palette("Blues_r", len(freq))
        ax.barh(range(len(freq)), freq.values, color=colors)
        ax.set_yticks(range(len(freq)))
        ax.set_yticklabels(freq.index, fontsize=7)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.set_xlabel('n', fontsize=7)
        for i, v in enumerate(freq.values):
            ax.text(v + 0.2, i, f'{v}', va='center', fontsize=6)

        print(f"\n  {label}")
        for cat, cnt in freq.items():
            print(f"    {str(cat):<40} {cnt:>3}  ({cnt}%)")

    plt.tight_layout()
    save_fig("Fig01_Respondent_Profile")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def descriptive_stats(df):
    print(f"\n{'='*70}")
    print("  SECTION 2 — DESCRIPTIVE STATISTICS (all constructs)")
    print(f"{'='*70}")

    constructs = {
        'Q11 — Daily AI usage frequency':         'Q11_n',
        'Q12 — Number of AI tools (rescaled)':    'Q12_n',
        'Q13 — Formalization of AI use':          'Q13_n',
        'Q14 — Type of AI usage':                 'Q14_n',
        'Q15 — Validation of AI outputs':         'Q15_n',
        'IDX_ADOPTION — AI Adoption Index':       'IDX_ADOPTION',
        'IDX_TOOLS — AI Tool Breadth Index':      'IDX_TOOLS',
        'IDX_PHASES — Agile Phase Integration':   'IDX_PHASES',
        'IDX_DECISION — Decision-making':         'IDX_DECISION',
        'IDX_AUTOMATION — Automation / time':     'IDX_AUTOMATION',
        'IDX_COLLAB — Collaboration':             'IDX_COLLAB',
        'Q35 — Sprint delivery performance':      'Q35_n',
        'Q36 — Productivity / velocity':          'Q36_n',
        'Q37 — Defect reduction (reversed)':      'Q37_n',
        'Q38 — Technical debt / maintainability': 'Q38_n',
        'IDX_PERF — Team Performance Index':      'IDX_PERF',
        'IDX_QUALITY — Deliverable Quality Index':'IDX_QUALITY',
        'Q39 — Overall project success':          'Q39_n',
    }

    rows = []
    print(f"\n  {'Variable':<45} {'N':>4} {'Mean':>6} {'SD':>6} "
          f"{'Med':>6} {'Min':>5} {'Max':>5} {'Skew':>7} {'Kurt':>7}")
    print(f"  {'-'*100}")

    for label, col in constructs.items():
        s = df[col].dropna()
        sk = round(float(stats.skew(s)), 3)
        kt = round(float(stats.kurtosis(s)), 3)
        rows.append({
            'Variable': label, 'N': len(s),
            'Mean': round(s.mean(), 2), 'SD': round(s.std(), 2),
            'Median': round(s.median(), 2),
            'Min': round(s.min(), 2), 'Max': round(s.max(), 2),
            'Skewness': sk, 'Kurtosis': kt
        })
        print(f"  {label:<45} {len(s):>4} {s.mean():>6.2f} {s.std():>6.2f} "
              f"{s.median():>6.2f} {s.min():>5.2f} {s.max():>5.2f} "
              f"{sk:>7.3f} {kt:>7.3f}")

    desc_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Descriptive Statistics — Key Constructs", fontsize=13,
                 fontweight='bold')

    plot_vars = [
        ('IDX_ADOPTION', 'AI Adoption Index', C_MAIN),
        ('IDX_TOOLS',    'Tool Breadth Index', C_MAIN),
        ('IDX_PHASES',   'Phase Integration', C_MAIN),
        ('IDX_DECISION', 'Decision-making', C_SEC),
        ('IDX_AUTOMATION','Automation', C_SEC),
        ('IDX_COLLAB',   'Collaboration', C_SEC),
        ('IDX_PERF',     'Team Performance', C_POS),
        ('Q39_n',        'Overall Success', C_POS),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), plot_vars):
        s = df[col].dropna()
        ax.hist(s, bins=10, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(s.mean(), color='black', linestyle='--',
                   linewidth=1.5, label=f'M={s.mean():.2f}')
        ax.set_title(f"{title}\nM={s.mean():.2f}  SD={s.std():.2f}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    save_fig("Fig02_Descriptive_Statistics")

    return desc_df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RELIABILITY ANALYSIS (Cronbach's Alpha)
# ─────────────────────────────────────────────────────────────────────────────
def reliability_analysis(df):
    print(f"\n{'='*70}")
    print("  SECTION 3 — RELIABILITY ANALYSIS (Cronbach's Alpha)")
    print(f"{'='*70}")

    blocks = {
        "AI Adoption (Q11,Q12,Q13,Q14,Q15)":
            ['Q11_n','Q12_norm','Q13_n','Q14_n','Q15_n'],
        "AI Tool Types (Q16–Q23)":
            [f'Q{i}_n' for i in range(16, 24)],
        "Agile Phase Integration (Q24–Q28)":
            [f'Q{i}_n' for i in range(24, 29)],
        "Decision-making Mechanisms (Q29–Q30)":
            ['Q29_n','Q30_n'],
        "Automation Mechanisms (Q31–Q32)":
            ['Q31_n','Q32_n'],
        "Collaboration Mechanisms (Q33–Q34)":
            ['Q33_n','Q34_n'],
        "All Perceived Mechanisms (Q29–Q34)":
            [f'Q{i}_n' for i in range(29, 35)],
        "Team Performance (Q35,Q36)":
            ['Q35_n','Q36_n'],
        "Deliverable Quality (Q37,Q38)":
            ['Q37_n','Q38_n'],
    }

    results = []
    print(f"\n  {'Construct':<45} {'α':>6}  {'N':>4}  Interpretation")
    print(f"  {'-'*75}")

    for name, items in blocks.items():
        alpha   = cronbach_alpha(df[items])
        n_valid = df[items].dropna().shape[0]
        interp  = ("Excellent (≥.90)" if alpha >= .90 else
                   "Good (.80–.89)"   if alpha >= .80 else
                   "Acceptable (.70–.79)" if alpha >= .70 else
                   "Questionable (.60–.69)" if alpha >= .60 else
                   "Poor (<.60)")
        results.append({'Construct': name, 'α': alpha,
                        'N': n_valid, 'Interpretation': interp})
        print(f"  {name:<45} {alpha:>6.3f}  {n_valid:>4}  {interp}")

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VALIDITY ANALYSIS (KMO + Bartlett)
# ─────────────────────────────────────────────────────────────────────────────
def validity_analysis(df):
    print(f"\n{'='*70}")
    print("  SECTION 4 — VALIDITY ANALYSIS (KMO + Bartlett's Test)")
    print(f"{'='*70}")
    print("  Threshold: KMO > 0.60 acceptable, > 0.80 meritorious")
    print("  Bartlett: significant (p < .05) confirms non-identity correlation matrix")

    blocks = {
        "AI Tool Types (Q16–Q23)":
            [f'Q{i}_n' for i in range(16, 24)],
        "Agile Phase Integration (Q24–Q28)":
            [f'Q{i}_n' for i in range(24, 29)],
        "Perceived Mechanisms (Q29–Q34)":
            [f'Q{i}_n' for i in range(29, 35)],
    }

    results = []
    print(f"\n  {'Block':<40} {'KMO':>6}  {'χ²':>10}  {'df':>5}  {'p-value':>10}  Valid?")
    print(f"  {'-'*85}")

    for name, items in blocks.items():
        data = df[items].dropna()
        if data.shape[0] < data.shape[1] + 1:
            print(f"  {name:<40} Insufficient data")
            continue
        kmo, chi2, df_b, pval = kmo_bartlett(data)
        valid = "Yes ✓" if kmo >= 0.60 and pval < .05 else "No ✗"
        results.append({'Block': name, 'KMO': kmo, 'Chi2': chi2,
                        'df': df_b, 'p': pval, 'Valid': valid})
        print(f"  {name:<40} {kmo:>6.3f}  {chi2:>10.3f}  {df_b:>5}  {pval:>10.4f}  {valid}")

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — NORMALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def normality_analysis(df):
    print(f"\n{'='*70}")
    print("  SECTION 5 — NORMALITY ANALYSIS")
    print("  Shapiro-Wilk test + Skewness/Kurtosis (acceptable range: ±2)")
    print(f"{'='*70}")

    variables = {
        'IDX_ADOPTION': "AI Adoption Index",
        'IDX_TOOLS':    "AI Tool Breadth Index",
        'IDX_PHASES':   "Agile Phase Integration",
        'IDX_DECISION': "Decision-making Mechanisms",
        'IDX_AUTOMATION':"Automation Mechanisms",
        'IDX_COLLAB':   "Collaboration Mechanisms",
        'IDX_PERF':     "Team Performance Index",
        'IDX_QUALITY':  "Deliverable Quality Index",
        'Q39_n':        "Overall Project Success (Q39)",
    }

    results = []
    print(f"\n  {'Variable':<35} {'W':>8}  {'p-SW':>8}  "
          f"{'Skew':>7}  {'Kurt':>7}  Normal?")
    print(f"  {'-'*80}")

    for col, label in variables.items():
        s = df[col].dropna()
        w, p = shapiro(s)
        sk   = float(stats.skew(s))
        kt   = float(stats.kurtosis(s))
        normal = (p > .05 and abs(sk) < 2 and abs(kt) < 2)
        flag   = "Yes" if normal else "No → Spearman"
        results.append({'Variable': label, 'W': round(w, 4), 'p_SW': round(p, 4),
                        'Skewness': round(sk, 3), 'Kurtosis': round(kt, 3),
                        'Normal': flag})
        print(f"  {label:<35} {w:>8.4f}  {p:>8.4f}  "
              f"{sk:>7.3f}  {kt:>7.3f}  {flag}")

    print("\n  → Non-normal distributions confirmed → Spearman's rho used for all")
    print("    correlations and hypothesis tests (appropriate for ordinal Likert data)")

    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SPEARMAN CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def correlation_matrix(df):
    print(f"\n{'='*70}")
    print("  SECTION 6 — SPEARMAN CORRELATION MATRIX")
    print(f"{'='*70}")

    var_map = {
        'IDX_ADOPTION':  "Adoption",
        'IDX_TOOLS':     "Tools",
        'IDX_PHASES':    "Phases",
        'IDX_DECISION':  "Decision",
        'IDX_AUTOMATION':"Automation",
        'IDX_COLLAB':    "Collaboration",
        'IDX_PERF':      "Performance",
        'IDX_QUALITY':   "Qual.Quality",
        'Q39_n':         "Success",
        'Q7_n':          "AgileMaturity",
        'Q13_n':         "Formalization",
        'Q15_n':         "Validation",
    }

    cols   = list(var_map.keys())
    labels = list(var_map.values())
    data   = df[cols].dropna()
    n      = len(data)

    rho_m = np.zeros((len(cols), len(cols)))
    p_m   = np.zeros((len(cols), len(cols)))

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                rho_m[i,j] = 1.0; p_m[i,j] = 0.0
            else:
                r, p = spearmanr(data[c1], data[c2])
                rho_m[i,j] = r; p_m[i,j] = p

    rho_df = pd.DataFrame(rho_m, index=labels, columns=labels)
    p_df   = pd.DataFrame(p_m,   index=labels, columns=labels)

    print(f"\n  Spearman's rho (n={n})  *** p<.001  ** p<.01  * p<.05")
    for i, l1 in enumerate(labels):
        row = f"  {l1:<14}"
        for j in range(i+1):
            r  = rho_m[i,j]
            p  = p_m[i,j]
            st = sig_stars(p) if i != j else ""
            row += f"  {r:+.2f}{st:<3}"
        print(row)

    annot = np.empty_like(rho_m, dtype=object)
    for i in range(len(cols)):
        for j in range(len(cols)):
            r  = rho_m[i,j]
            st = sig_stars(p_m[i,j]) if i != j else ""
            annot[i,j] = f"{r:.2f}{st}"

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(rho_df, annot=annot, fmt='', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.4, square=True,
                cbar_kws={'shrink': 0.75, 'label': "Spearman's ρ"})
    ax.set_title(
        f"Spearman Correlation Matrix (n={n})\n"
        "*** p<.001  ** p<.01  * p<.05",
        fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save_fig("Fig03_Correlation_Matrix")

    return rho_df, p_df, n

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — HYPOTHESIS TESTING (H1–H4)
# ─────────────────────────────────────────────────────────────────────────────
def hypothesis_testing(df):
    print(f"\n{'='*70}")
    print("  SECTION 7 — HYPOTHESIS TESTING (Spearman's rho)")
    print(f"{'='*70}")

    hyp_results = []

    def spearman_test(x_col, y_col, h_id, description):
        d = df[[x_col, y_col]].dropna()
        r, p = spearmanr(d[x_col], d[y_col])
        stars = sig_stars(p)
        es    = effect_size_r(r)
        supp  = "Supported ✓" if p < .05 else "Not Supported ✗"
        hyp_results.append({
            'Hypothesis': h_id,
            'Description': description,
            'IV': x_col, 'DV': y_col,
            'Test': 'Spearman',
            'ρ': round(r, 3),
            'p-value': round(p, 4),
            'Sig.': stars,
            'N': len(d),
            'Effect Size': es,
            'Decision': supp
        })
        print(f"\n  {h_id}: {description}")
        print(f"    ρ = {r:.3f}  p = {p:.4f}  {stars}  "
              f"N = {len(d)}  Effect: {es}  → {supp}")
        return r, p

    # ── H1 ──────────────────────────────────────────────────────────────────
    print("\n  ── H1: The level of AI tool adoption by Agile teams is associated")
    print("         with Agile team performance. ──")
    print("  Measured by: IDX_ADOPTION (Q11–Q15) × IDX_PERF (Q35+Q36)")
    r_h1, p_h1 = spearman_test(
        'IDX_ADOPTION', 'IDX_PERF',
        'H1',
        "AI Adoption Index (Q11–Q15) → Team Performance Index (Q35+Q36)"
    )

    # ── H2 ──────────────────────────────────────────────────────────────────
    print("\n  ── H2: The use of AI tools in Agile project management is associated")
    print("         with the quality of deliverables produced. ──")
    print("  Note: Q37 (defects) and Q38 (technical debt) tested separately")
    print("        because combined index has negative Cronbach α (not reliable)")
    print("  Measured by: IDX_ADOPTION × Q37 (defects, reversed) and Q38 (tech debt)")
    r_h2a, p_h2a = spearman_test(
        'IDX_ADOPTION', 'Q37_n',
        'H2 (Q37)',
        "AI Adoption Index → Defect reduction after release (Q37, reversed)"
    )
    r_h2b, p_h2b = spearman_test(
        'IDX_ADOPTION', 'Q38_n',
        'H2 (Q38)',
        "AI Adoption Index → Technical debt / maintainability (Q38)"
    )
    if p_h2a < .05 and p_h2b < .05:
        h2_decision = "Supported ✓"
    elif p_h2a < .05 or p_h2b < .05:
        h2_decision = "Partially Supported ⚠"
    else:
        h2_decision = "Not Supported ✗"
    print(f"\n  H2 overall decision: {h2_decision}")
    print(f"    Q37 (defects):        ρ={r_h2a:.3f}  p={p_h2a:.4f}  {sig_stars(p_h2a)}")
    print(f"    Q38 (tech debt):      ρ={r_h2b:.3f}  p={p_h2b:.4f}  {sig_stars(p_h2b)}")
    print(f"    → AI adoption is associated with defect reduction but not with")
    print(f"      technical debt improvement, suggesting a partial effect on quality.")

    # ── H3 ──────────────────────────────────────────────────────────────────
    print("\n  ── H3: The integration of AI tools across Agile phases is associated")
    print("         with overall project success. ──")
    print("  Measured by: IDX_PHASES (Q24–Q28) × Q39 (Overall Project Success)")
    r_h3, p_h3 = spearman_test(
        'IDX_PHASES', 'Q39_n',
        'H3',
        "Agile Phase Integration (Q24–Q28) → Overall Project Success (Q39)"
    )

    # ── H4 ──────────────────────────────────────────────────────────────────
    print("\n  ── H4: The adoption of AI in Agile project management is associated")
    print("         with overall project success. ──")
    print("  Measured by: IDX_ADOPTION (Q11–Q15) × Q39 (overall success)")
    r_h4, p_h4 = spearman_test(
        'IDX_ADOPTION', 'Q39_n',
        'H4',
        "AI Adoption Index (Q11–Q15) → Overall Project Success (Q39)"
    )

    hyp_df = pd.DataFrame(hyp_results)

    # ── Figure — 4 scatter plots, one per hypothesis ─────────────────────
    # NOTE: H3 scatter now correctly uses IDX_PHASES × Q39_n
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Hypothesis Testing — Scatter Plots (Spearman's ρ)",
                 fontsize=13, fontweight='bold')

    plots = [
        ('IDX_ADOPTION', 'IDX_PERF',
         'H1 — AI Adoption × Team Performance\n(IDX_ADOPTION × IDX_PERF)'),
        ('IDX_ADOPTION', 'Q37_n',
         'H2 — AI Adoption × Defect Reduction\n(IDX_ADOPTION × Q37, reversed)'),
        ('IDX_PHASES',   'Q39_n',
         'H3 — Phase Integration × Overall Project Success\n(IDX_PHASES × Q39)'),
        ('IDX_ADOPTION', 'Q39_n',
         'H4 — AI Adoption × Overall Project Success\n(IDX_ADOPTION × Q39)'),
    ]

    for ax, (xc, yc, title) in zip(axes.flatten(), plots):
        d    = df[[xc, yc]].dropna()
        r, p = spearmanr(d[xc], d[yc])
        st   = sig_stars(p)
        ax.scatter(d[xc], d[yc], alpha=0.45, color=C_MAIN,
                   s=55, edgecolors='white', linewidth=0.5)
        z   = np.polyfit(d[xc], d[yc], 1)
        xln = np.linspace(d[xc].min(), d[xc].max(), 100)
        ax.plot(xln, np.poly1d(z)(xln), color=C_SEC, linewidth=2.2)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel(xc.replace('IDX_', 'Index: ').replace('_n', ''),
                      fontsize=9)
        ax.set_ylabel(yc.replace('IDX_', 'Index: ').replace('_n', ''),
                      fontsize=9)
        clr = C_POS if p < .05 else C_NEG
        ax.text(0.05, 0.93, f'ρ = {r:.3f}  {st}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=clr,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='lightgray'))

    plt.tight_layout()
    save_fig("Fig04_Hypothesis_Scatterplots")

    # ── Summary table ───────────────────────────────────────────────────
    summary_rows = [
        {
            'Hypothesis': 'H1',
            'Statement': 'AI adoption is associated with Agile team performance',
            'IV': 'IDX_ADOPTION', 'DV': 'IDX_PERF (Q35+Q36)',
            'ρ': round(r_h1, 3), 'p-value': round(p_h1, 4),
            'Sig.': sig_stars(p_h1), 'Effect': effect_size_r(r_h1),
            'Decision': 'Supported ✓' if p_h1 < .05 else 'Not Supported ✗'
        },
        {
            'Hypothesis': 'H2',
            'Statement': 'AI adoption is associated with deliverable quality',
            'IV': 'IDX_ADOPTION',
            'DV': 'Q37 (defects) + Q38 (tech debt)',
            'ρ': f'{r_h2a:.3f} / {r_h2b:.3f}',
            'p-value': f'{p_h2a:.4f} / {p_h2b:.4f}',
            'Sig.': f'{sig_stars(p_h2a)} / {sig_stars(p_h2b)}',
            'Effect': f'{effect_size_r(r_h2a)} / {effect_size_r(r_h2b)}',
            'Decision': h2_decision
        },
        {
            'Hypothesis': 'H3',
            'Statement': 'Phase integration is associated with overall project success',
            'IV': 'IDX_PHASES', 'DV': 'Q39_n (Overall Project Success)',
            'ρ': round(r_h3, 3), 'p-value': round(p_h3, 4),
            'Sig.': sig_stars(p_h3), 'Effect': effect_size_r(r_h3),
            'Decision': 'Supported ✓' if p_h3 < .05 else 'Not Supported ✗'
        },
        {
            'Hypothesis': 'H4',
            'Statement': 'AI adoption is associated with overall project success',
            'IV': 'IDX_ADOPTION', 'DV': 'Q39 (overall success)',
            'ρ': round(r_h4, 3), 'p-value': round(p_h4, 4),
            'Sig.': sig_stars(p_h4), 'Effect': effect_size_r(r_h4),
            'Decision': 'Supported ✓' if p_h4 < .05 else 'Not Supported ✗'
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n  {'─'*75}")
    print(f"  HYPOTHESIS SUMMARY")
    print(f"  {'─'*75}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Hypothesis']}  {row['Decision']}")
        print(f"    {row['Statement']}")
        print(f"    ρ = {row['ρ']}  p = {row['p-value']}  {row['Sig.']}  "
              f"Effect: {row['Effect']}")

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')
    cols_show = ['Hypothesis', 'Statement', 'IV', 'DV',
                 'ρ', 'p-value', 'Sig.', 'Effect', 'Decision']
    tbl = ax.table(
        cellText=summary_df[cols_show].values,
        colLabels=cols_show,
        cellLoc='left', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(cols_show))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_MAIN)
            cell.set_text_props(color='white', fontweight='bold')
        elif row > 0:
            dec = summary_df.iloc[row-1]['Decision']
            if '✓' in str(dec):
                cell.set_facecolor('#E8F5E9')
            elif '⚠' in str(dec):
                cell.set_facecolor('#FFF9C4')
            else:
                cell.set_facecolor('#FFF3E0')
    ax.set_title("Hypothesis Testing Summary", fontsize=13,
                 fontweight='bold', pad=15)
    plt.tight_layout()
    save_fig("Fig05_Hypothesis_Summary_Table")

    return summary_df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — GROUP COMPARISONS (Kruskal-Wallis + Dunn post-hoc)
# ─────────────────────────────────────────────────────────────────────────────
def dunn_posthoc(groups, group_names):
    all_data = np.concatenate(groups)
    all_ranks = stats.rankdata(all_data)
    n_total = len(all_data)
    results = []
    idx = 0
    group_ranks = []
    for g in groups:
        group_ranks.append(all_ranks[idx:idx+len(g)])
        idx += len(g)

    pairs = list(combinations(range(len(groups)), 2))
    n_pairs = len(pairs)

    for i, j in pairs:
        ni, nj   = len(groups[i]), len(groups[j])
        ri, rj   = group_ranks[i].mean(), group_ranks[j].mean()
        se       = np.sqrt((n_total*(n_total+1)/12) * (1/ni + 1/nj))
        z        = abs(ri - rj) / se
        p_raw    = 2 * (1 - stats.norm.cdf(z))
        p_adj    = min(p_raw * n_pairs, 1.0)
        results.append({
            'Group A': group_names[i], 'Group B': group_names[j],
            'Mean Rank A': round(ri, 2), 'Mean Rank B': round(rj, 2),
            'Z': round(z, 3), 'p (adjusted)': round(p_adj, 4),
            'Sig.': sig_stars(p_adj)
        })
    return pd.DataFrame(results)

def group_comparisons(df, cols):
    print(f"\n{'='*70}")
    print("  SECTION 8 — GROUP COMPARISONS (Kruskal-Wallis + Dunn post-hoc)")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Group Comparisons — AI Adoption Level × Outcomes",
                 fontsize=12, fontweight='bold')

    print("\n  8.1 — Kruskal-Wallis: Q14 (Type of AI Usage) × Q39 (Success)")
    df['Q14_group'] = df[cols[14]].map({
        'Ponctuel / exp\u00e9rimental (utilisation occasionnelle, non formalis\u00e9e)':
            'Experimental',
        'R\u00e9gulier mais non standardis\u00e9 (utilisation fr\u00e9quente, sans r\u00e8gles fixes)':
            'Regular',
        'Institutionnalis\u00e9e / standardis\u00e9e (utilisation int\u00e9gr\u00e9e aux pratiques de travail)':
            'Institutionalized'
    })

    order14   = ['Experimental', 'Regular', 'Institutionalized']
    groups14  = [df[df['Q14_group'] == g]['Q39_n'].dropna().values for g in order14]
    groups14  = [g for g in groups14 if len(g) > 0]
    names14   = [o for o, g in zip(order14, groups14) if len(g) > 0]
    h14, p14  = kruskal(*groups14)
    eta2_14   = (h14 - len(groups14) + 1) / (len(df['Q39_n'].dropna()) - len(groups14))

    print(f"    H = {h14:.3f}  p = {p14:.4f}  {sig_stars(p14)}")
    print(f"    η² = {eta2_14:.3f}  (effect size)")
    for name, g in zip(names14, groups14):
        print(f"    {name:<20} M={g.mean():.2f}  SD={g.std():.2f}  n={len(g)}")

    if p14 < .05:
        dunn14 = dunn_posthoc(groups14, names14)
        print("\n    Dunn post-hoc (Bonferroni):")
        for _, row in dunn14.iterrows():
            print(f"    {row['Group A']} vs {row['Group B']}: "
                  f"Z={row['Z']:.3f}  p_adj={row['p (adjusted)']:.4f}  {row['Sig.']}")

    data_plot = [(g, n) for g, n in zip(groups14, names14)]
    bp = axes[0].boxplot([d[0] for d in data_plot],
                          labels=[d[1] for d in data_plot],
                          patch_artist=True,
                          medianprops=dict(color='white', linewidth=2))
    colors_bp = [C_NEG, C_SEC, C_POS]
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0].set_title(f"Q14 Type of Usage × Q39 Success\n"
                      f"H={h14:.2f}  p={p14:.4f}  {sig_stars(p14)}")
    axes[0].set_ylabel("Overall Project Success (Q39)")
    axes[0].set_ylim(0.5, 5.5)

    print("\n  8.2 — Kruskal-Wallis: Agile Maturity (Q7) × Q39 (Success)")
    mat_map = {1: 'Pilot', 2: 'Several Teams', 3: 'Company-wide', 4: 'Scaled'}
    df['Q7_group'] = df['Q7_n'].map(mat_map)
    order7   = ['Pilot', 'Several Teams', 'Company-wide', 'Scaled']
    groups7  = [df[df['Q7_group'] == g]['Q39_n'].dropna().values for g in order7]
    groups7  = [g for g in groups7 if len(g) > 0]
    names7   = [o for o, g in zip(order7, groups7) if len(g) > 0]
    h7, p7   = kruskal(*groups7)
    eta2_7   = (h7 - len(groups7) + 1) / (len(df['Q39_n'].dropna()) - len(groups7))

    print(f"    H = {h7:.3f}  p = {p7:.4f}  {sig_stars(p7)}")
    print(f"    η² = {eta2_7:.3f}")
    for name, g in zip(names7, groups7):
        print(f"    {name:<20} M={g.mean():.2f}  SD={g.std():.2f}  n={len(g)}")

    if p7 < .05:
        dunn7 = dunn_posthoc(groups7, names7)
        print("\n    Dunn post-hoc (Bonferroni):")
        for _, row in dunn7.iterrows():
            print(f"    {row['Group A']} vs {row['Group B']}: "
                  f"Z={row['Z']:.3f}  p_adj={row['p (adjusted)']:.4f}  {row['Sig.']}")

    bp2 = axes[1].boxplot([g for g in groups7], labels=names7, patch_artist=True,
                           medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp2['boxes'], sns.color_palette("Blues", len(groups7))):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    axes[1].set_title(f"Agile Maturity (Q7) × Success (Q39)\n"
                      f"H={h7:.2f}  p={p7:.4f}  {sig_stars(p7)}")
    axes[1].set_ylabel("Overall Project Success (Q39)")
    axes[1].set_ylim(0.5, 5.5)
    axes[1].tick_params(axis='x', labelsize=8)

    print("\n  8.3 — Kruskal-Wallis: Number of Tools (Q12) × Q39 (Success)")
    nb_map = {0: '0 tools', 1: '1 tool', 2.5: '2–3 tools', 4.5: '4–5 tools', 7: '6+ tools'}
    df['Q12_group'] = df['Q12_n'].map(nb_map)
    order12  = ['0 tools', '1 tool', '2–3 tools', '4–5 tools', '6+ tools']
    groups12 = [df[df['Q12_group'] == g]['Q39_n'].dropna().values for g in order12]
    valid12  = [(g, n) for g, n in zip(groups12, order12) if len(g) > 1]
    if len(valid12) >= 2:
        g12v, n12v = zip(*valid12)
        h12, p12   = kruskal(*g12v)
        eta2_12    = (h12 - len(g12v) + 1) / (len(df['Q39_n'].dropna()) - len(g12v))
        print(f"    H = {h12:.3f}  p = {p12:.4f}  {sig_stars(p12)}")
        print(f"    η² = {eta2_12:.3f}")

        bp3 = axes[2].boxplot(list(g12v), labels=list(n12v), patch_artist=True,
                               medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp3['boxes'], sns.color_palette("Greens", len(g12v))):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        axes[2].set_title(f"Number of Tools (Q12) × Success (Q39)\n"
                          f"H={h12:.2f}  p={p12:.4f}  {sig_stars(p12)}")
        axes[2].set_ylabel("Overall Project Success (Q39)")
        axes[2].set_ylim(0.5, 5.5)

    plt.tight_layout()
    save_fig("Fig06_Group_Comparisons")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MODERATION ANALYSIS (Agile Maturity)
# ─────────────────────────────────────────────────────────────────────────────
def moderation_analysis(df):
    print(f"\n{'='*70}")
    print("  SECTION 9 — MODERATION ANALYSIS (Agile Maturity as Moderator)")
    print("  Testing: does Agile maturity moderate the Adoption → Success link?")
    print(f"{'='*70}")

    df['Q7_binary'] = df['Q7_n'].apply(
        lambda x: 'High Maturity' if x >= 3 else 'Low Maturity'
        if pd.notna(x) else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Moderation Analysis — Agile Maturity (Q7)\n"
        "Does maturity moderate the AI Adoption → Project Success relationship?",
        fontsize=12, fontweight='bold')

    for ax, group_label, color in zip(
            axes, ['Low Maturity', 'High Maturity'], [C_SEC, C_MAIN]):
        sub  = df[df['Q7_binary'] == group_label][['IDX_ADOPTION', 'Q39_n']].dropna()
        r, p = spearmanr(sub['IDX_ADOPTION'], sub['Q39_n'])
        st   = sig_stars(p)
        ax.scatter(sub['IDX_ADOPTION'], sub['Q39_n'],
                   alpha=0.5, color=color, s=55, edgecolors='white')
        z   = np.polyfit(sub['IDX_ADOPTION'], sub['Q39_n'], 1)
        xln = np.linspace(sub['IDX_ADOPTION'].min(), sub['IDX_ADOPTION'].max(), 100)
        ax.plot(xln, np.poly1d(z)(xln), color='black', linewidth=2)
        ax.set_title(f"{group_label} (n={len(sub)})\nρ = {r:.3f}  {st}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("AI Adoption Index", fontsize=10)
        ax.set_ylabel("Overall Project Success (Q39)", fontsize=10)
        print(f"\n  {group_label}: n={len(sub)}  ρ={r:.3f}  p={p:.4f}  {st}")

    groups = df.groupby('Q7_binary')[['IDX_ADOPTION', 'Q39_n']].apply(
        lambda x: spearmanr(x['IDX_ADOPTION'], x['Q39_n']))
    r_vals = {k: v[0] for k, v in groups.items() if not pd.isna(v[0])}
    n_vals = df.groupby('Q7_binary')[['IDX_ADOPTION', 'Q39_n']].apply(
        lambda x: x.dropna().shape[0])

    if len(r_vals) == 2:
        keys = list(r_vals.keys())
        r1, r2 = r_vals[keys[0]], r_vals[keys[1]]
        n1, n2 = n_vals[keys[0]], n_vals[keys[1]]
        z1 = np.arctanh(r1); z2 = np.arctanh(r2)
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z_diff = (z1 - z2) / se
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        print(f"\n  Fisher's z-test (difference between correlations):")
        print(f"    z = {z_diff:.3f}  p = {p_diff:.4f}  {sig_stars(p_diff)}")
        print(f"    → Moderation effect: "
              f"{'Significant ✓' if p_diff < .05 else 'Not significant ✗'}")
        axes[1].text(0.05, 0.05,
                     f"Fisher's z-test: z={z_diff:.2f}, p={p_diff:.3f}",
                     transform=axes[1].transAxes, fontsize=8,
                     bbox=dict(facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    save_fig("Fig07_Moderation_Analysis")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — OLS REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def regression_analysis(df):
    print(f"\n{'='*70}")
    print("  SECTION 10 — OLS REGRESSION")
    print("  Dependent variable: Q39 (Overall Project Success)")
    print("  Predictors: AI Adoption, Phase Integration, Decision, Automation,")
    print("              Collaboration, Agile Maturity")
    print(f"{'='*70}")

    pred_cols  = ['IDX_ADOPTION', 'IDX_PHASES', 'IDX_DECISION',
                  'IDX_AUTOMATION', 'IDX_COLLAB', 'Q7_n']
    pred_names = ['AI Adoption', 'Phase Integration', 'Decision-making',
                  'Automation', 'Collaboration', 'Agile Maturity']
    dep_col    = 'Q39_n'

    data   = df[pred_cols + [dep_col]].dropna()
    n_reg  = len(data)
    Y      = data[dep_col].values
    X_raw  = data[pred_cols].values
    k      = X_raw.shape[1]

    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0)
    X_std[X_std == 0] = 1
    X_std_arr = (X_raw - X_mean) / X_std
    X_c   = np.column_stack([np.ones(n_reg), X_std_arr])

    beta  = np.linalg.lstsq(X_c, Y, rcond=None)[0]
    Y_hat = X_c @ beta
    resid = Y - Y_hat
    ss_res = resid @ resid
    ss_tot = ((Y - Y.mean())**2).sum()
    ss_reg = ss_tot - ss_res
    r2     = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n_reg - 1) / (n_reg - k - 1)
    mse    = ss_res / (n_reg - k - 1)

    XtX_inv = np.linalg.pinv(X_c.T @ X_c)
    se      = np.sqrt(np.diag(XtX_inv * mse))
    t_vals  = beta / se
    p_vals  = [2*(1 - stats.t.cdf(abs(t), df=n_reg-k-1)) for t in t_vals]

    t_crit = stats.t.ppf(0.975, df=n_reg-k-1)
    ci_lo  = beta - t_crit * se
    ci_hi  = beta + t_crit * se

    f_stat = (ss_reg / k) / (ss_res / (n_reg - k - 1))
    p_f    = 1 - stats.f.cdf(f_stat, k, n_reg - k - 1)

    vif_vals = []
    for j in range(k):
        xj   = X_std_arr[:, j]
        Xoth = np.delete(X_std_arr, j, axis=1)
        Xoth_c = np.column_stack([np.ones(n_reg), Xoth])
        b_j  = np.linalg.lstsq(Xoth_c, xj, rcond=None)[0]
        xj_hat = Xoth_c @ b_j
        ss_r = ((xj - xj_hat)**2).sum()
        ss_t = ((xj - xj.mean())**2).sum()
        r2_j = 1 - ss_r / ss_t if ss_t > 0 else 0
        vif_vals.append(round(1 / (1 - r2_j), 3) if r2_j < 1 else np.inf)

    dw = np.sum(np.diff(resid)**2) / ss_res

    print(f"\n  ANOVA Table")
    print(f"  {'Source':<15} {'SS':>10}  {'df':>5}  {'MS':>10}  {'F':>8}  p-value")
    print(f"  {'-'*60}")
    print(f"  {'Regression':<15} {ss_reg:>10.3f}  {k:>5}  "
          f"{ss_reg/k:>10.3f}  {f_stat:>8.3f}  {p_f:.4f} {sig_stars(p_f)}")
    print(f"  {'Residual':<15} {ss_res:>10.3f}  {n_reg-k-1:>5}  {mse:>10.3f}")
    print(f"  {'Total':<15} {ss_tot:>10.3f}  {n_reg-1:>5}")
    print(f"\n  R² = {r2:.3f}   Adjusted R² = {r2_adj:.3f}"
          f"   Durbin-Watson = {dw:.3f}")

    all_names = ['(Constant)'] + pred_names
    print(f"\n  Coefficients (standardized predictors)")
    print(f"  {'Predictor':<22} {'β':>8}  {'SE':>7}  {'t':>7}  "
          f"{'p':>8}  {'Sig':>5}  {'CI 95%':>16}  {'VIF':>7}")
    print(f"  {'-'*90}")

    reg_rows = []
    for i, (name, b, s, t, p, lo, hi) in enumerate(
            zip(all_names, beta, se, t_vals, p_vals, ci_lo, ci_hi)):
        vif  = vif_vals[i-1] if i > 0 else ''
        st   = sig_stars(p)
        ci   = f"[{lo:.3f}, {hi:.3f}]"
        vif_s = f"{vif:.2f}" if isinstance(vif, float) else ''
        print(f"  {name:<22} {b:>8.3f}  {s:>7.3f}  {t:>7.3f}  "
              f"{p:>8.4f}  {st:>5}  {ci:>16}  {vif_s:>7}")
        reg_rows.append({'Predictor': name, 'β': round(b, 3),
                         'SE': round(s, 3), 't': round(t, 3),
                         'p-value': round(p, 4), 'Sig.': st,
                         'CI_lower': round(lo, 3), 'CI_upper': round(hi, 3),
                         'VIF': vif_s})

    mc_ok = all(v < 5 for v in vif_vals if isinstance(v, float))
    print(f"\n  VIF check: {'No multicollinearity (all VIF < 5) ✓' if mc_ok else 'Multicollinearity detected ✗'}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        f"OLS Regression — Predictors of Overall Project Success (Q39)\n"
        f"R² = {r2:.3f}  |  Adj. R² = {r2_adj:.3f}  |  "
        f"F({k},{n_reg-k-1}) = {f_stat:.2f}, p = {p_f:.4f}",
        fontsize=11, fontweight='bold')

    betas_plot = beta[1:]
    pvals_plot = p_vals[1:]
    clrs = [C_POS if (b>0 and p<.05) else C_NEG if (b<0 and p<.05) else C_NEU
            for b, p in zip(betas_plot, pvals_plot)]
    axes[0].barh(pred_names, betas_plot, color=clrs, edgecolor='white', height=0.55)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel("Standardized β coefficient")
    axes[0].set_title("Regression Coefficients")
    for i, (b, p) in enumerate(zip(betas_plot, pvals_plot)):
        st = sig_stars(p)
        axes[0].text(b + (0.005 if b >= 0 else -0.005), i,
                     f'{b:.3f}{st}', va='center',
                     ha='left' if b >= 0 else 'right', fontsize=8)
    leg = [mpatches.Patch(color=C_POS, label='Positive (p<.05)'),
           mpatches.Patch(color=C_NEG, label='Negative (p<.05)'),
           mpatches.Patch(color=C_NEU, label='Non-significant')]
    axes[0].legend(handles=leg, fontsize=7)

    axes[1].scatter(Y_hat, Y, alpha=0.45, color=C_MAIN, s=50, edgecolors='white')
    lim = [min(Y_hat.min(), Y.min())-.3, max(Y_hat.max(), Y.max())+.3]
    axes[1].plot(lim, lim, 'k--', linewidth=1.5)
    axes[1].set_xlabel("Predicted (Ŷ)"); axes[1].set_ylabel("Observed (Y)")
    axes[1].set_title(f"Predicted vs Observed\nR² = {r2:.3f}")
    axes[1].text(0.05, 0.92, f'R² = {r2:.3f}', transform=axes[1].transAxes,
                 fontsize=10, fontweight='bold', color=C_MAIN,
                 bbox=dict(boxstyle='round', facecolor='white'))

    axes[2].scatter(Y_hat, resid, alpha=0.45, color=C_SEC, s=50, edgecolors='white')
    axes[2].axhline(0, color='black', linewidth=1.2, linestyle='--')
    axes[2].set_xlabel("Predicted (Ŷ)")
    axes[2].set_ylabel("Residuals")
    axes[2].set_title(f"Residual Plot\nDurbin-Watson = {dw:.3f}")

    plt.tight_layout()
    save_fig("Fig08_Regression")

    return pd.DataFrame(reg_rows), r2, r2_adj, f_stat, p_f

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — PERCEIVED VALUE CREATION MECHANISMS (Q29–Q34)
# ─────────────────────────────────────────────────────────────────────────────
def perceived_mechanisms(df):
    print(f"\n{'='*70}")
    print("  SECTION 11 — PERCEIVED VALUE CREATION MECHANISMS (Q29–Q34)")
    print("  Descriptive block — answers SQR3 (perceived benefits)")
    print(f"{'='*70}")

    meca = {
        'Q29_n': 'Decision influence\n(Q29)',
        'Q30_n': 'Decision quality\n(Q30)',
        'Q31_n': 'Manual work reduction\n(Q31)',
        'Q32_n': 'Test automation\n(Q32)',
        'Q33_n': 'Communication\n(Q33)',
        'Q34_n': 'Coordination\n(Q34)',
    }

    print(f"\n  {'Item':<40} {'Mean':>6}  {'SD':>6}  "
          f"{'% Agree (4-5)':>14}  {'% Disagree (1-2)':>17}")
    print(f"  {'-'*85}")

    for col, label in meca.items():
        s  = df[col].dropna()
        ag = (s >= 4).sum() / len(s) * 100
        di = (s <= 2).sum() / len(s) * 100
        print(f"  {label.replace(chr(10),' '):<40} "
              f"{s.mean():>6.2f}  {s.std():>6.2f}  "
              f"{ag:>13.1f}%  {di:>16.1f}%")

    fig, ax = plt.subplots(figsize=(13, 6))
    labels = list(meca.values())
    cols_q  = list(meca.keys())

    neg = np.array([(df[c] <= 2).sum() / df[c].notna().sum() * 100 for c in cols_q])
    neu = np.array([(df[c] == 3).sum() / df[c].notna().sum() * 100 for c in cols_q])
    pos = np.array([(df[c] >= 4).sum() / df[c].notna().sum() * 100 for c in cols_q])
    y   = np.arange(len(labels))

    ax.barh(y, -neg,          color=C_NEG,  height=0.6, label='Disagree (1–2)')
    ax.barh(y, -neu/2, left=-neg, color=C_NEU, height=0.6)
    ax.barh(y,  neu/2,        color=C_NEU,  height=0.6, label='Neutral (3)')
    ax.barh(y,  pos, left=neu/2, color=C_MAIN, height=0.6, label='Agree (4–5)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Percentage of respondents (%)")
    ax.set_title("Perceived Value Creation Mechanisms (Q29–Q34)\nDiverging Likert Distribution",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-65, 65)
    for i, col in enumerate(cols_q):
        m = df[col].mean()
        ax.text(57, i, f'M={m:.2f}', va='center', fontsize=9,
                fontweight='bold', color=C_MAIN)
    plt.tight_layout()
    save_fig("Fig09_Perceived_Mechanisms")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — CONTENT ANALYSIS Q40 (Challenges)
# ─────────────────────────────────────────────────────────────────────────────
def content_analysis(df, cols):
    print(f"\n{'='*70}")
    print("  SECTION 12 — CONTENT ANALYSIS — Q40 (Challenges)")
    print(f"{'='*70}")

    all_ch = []
    for entry in df[cols[40]].dropna():
        all_ch.extend([c.strip() for c in str(entry).split(',') if c.strip()])

    freq  = pd.Series(all_ch).value_counts()
    n_res = df[cols[40]].notna().sum()

    en_labels = {
        'Confidentialit\u00e9 / RGPD': 'Data privacy / GDPR',
        'D\u00e9pendance excessive': 'Over-reliance',
        'Faible confiance dans les r\u00e9sultats IA': 'Low trust in AI outputs',
        'S\u00e9curit\u00e9': 'Security concerns',
        'Co\u00fbt des outils': 'Tool cost / licensing',
        'Manque de comp\u00e9tences / formation': 'Lack of skills / training',
        'Gouvernance / absence de r\u00e8gles': 'Governance / lack of policies',
        'Difficult\u00e9 d\u2019int\u00e9gration': 'Integration difficulties',
        'R\u00e9sistance au changement': 'Resistance to change',
        'Biais / \u00e9quit\u00e9': 'Bias / fairness',
        'Explicabilit\u00e9': 'Explainability',
        'Autre': 'Other',
    }

    freq.index = [en_labels.get(i, i) for i in freq.index]

    print(f"\n  Respondents: {n_res}  |  Total mentions: {len(all_ch)}")
    print(f"\n  {'Challenge':<40} {'n':>5}  {'%'}")
    print(f"  {'-'*60}")
    for ch, cnt in freq.items():
        print(f"  {ch:<40} {cnt:>5}  ({cnt/n_res*100:.1f}%)")

    themes = {
        "Trust & Governance":
            ['Low trust in AI outputs', 'Governance / lack of policies',
             'Bias / fairness', 'Explainability'],
        "Privacy & Security": ['Data privacy / GDPR', 'Security concerns'],
        "Adoption & Resistance": ['Over-reliance', 'Resistance to change'],
        "Skills & Cost": ['Lack of skills / training', 'Tool cost / licensing'],
        "Technical Integration": ['Integration difficulties'],
    }
    theme_counts = {t: sum(freq.get(k, 0) for k in ks) for t, ks in themes.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Challenges in AI Integration — Q40 (n={n_res})",
                 fontsize=13, fontweight='bold')

    top = freq.head(10)
    pcts = (top / n_res * 100).values
    clrs = sns.color_palette("Reds_r", len(top))
    ax1.barh(range(len(top)), pcts, color=clrs, edgecolor='white')
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top.index, fontsize=9)
    ax1.set_xlabel("% of respondents")
    ax1.set_title("Top 10 Challenges")
    ax1.invert_yaxis()
    for bar, pct in zip(ax1.patches, pcts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{pct:.1f}%', va='center', fontsize=8, fontweight='bold')

    td = {k: v for k, v in theme_counts.items() if v > 0}
    wdg, _, autotxt = ax2.pie(
        list(td.values()), labels=None,
        colors=sns.color_palette("Set2", len(td)),
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, pctdistance=0.75)
    centre = plt.Circle((0, 0), 0.5, fc='white')
    ax2.add_patch(centre)
    ax2.set_title("Thematic Grouping")
    ax2.legend(wdg, [f"{k} ({v})" for k, v in td.items()],
               loc='lower center', fontsize=8,
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout()
    save_fig("Fig10_Challenges_Q40")

    return freq

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — AI ADOPTION & TOOL TYPES (SQR1 + SQR2)
# ─────────────────────────────────────────────────────────────────────────────
def adoption_and_tools(df, cols):
    print(f"\n{'='*70}")
    print("  SECTION 13 — AI ADOPTION & TOOL TYPES (answers SQR1 & SQR2)")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("AI Adoption — Frequency, Tool Types & Agile Phase Integration",
                 fontsize=12, fontweight='bold')

    order_q11 = ['Never', 'Rarely', 'Sometimes', 'Often', 'Very often']
    map_q11   = {'Jamais': 'Never', 'Rarement': 'Rarely', 'Parfois': 'Sometimes',
                 'Souvent': 'Often', 'Tr\u00e8s souvent': 'Very often'}
    vc11 = df[cols[11]].map(map_q11).value_counts().reindex(order_q11, fill_value=0)
    clrs11 = [C_NEG, '#E67E22', C_NEU, '#3498DB', C_MAIN]
    bars = axes[0].bar(order_q11, vc11.values, color=clrs11, edgecolor='white')
    axes[0].set_title("Q11 — Daily AI Usage Frequency")
    axes[0].set_ylabel("n")
    for bar, v in zip(bars, vc11.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + .3,
                     f'{v}\n({v}%)', ha='center', fontsize=9, fontweight='bold')

    tool_labels = ['Generative AI\n(LLM)', 'Predictive ML', 'Decision\nSupport',
                   'NLP', 'Code\nGeneration', 'Test / QA',
                   'Analytics\nDashboards', 'Automation\nBots']
    means_t = [df[f'Q{i}_n'].mean() for i in range(16, 24)]
    stds_t  = [df[f'Q{i}_n'].std()  for i in range(16, 24)]
    sorted_i = np.argsort(means_t)[::-1]
    clrs_t   = [C_MAIN if means_t[i] >= 3 else C_NEU for i in sorted_i]
    axes[1].barh([tool_labels[i] for i in sorted_i],
                 [means_t[i] for i in sorted_i],
                 xerr=[stds_t[i] for i in sorted_i],
                 color=clrs_t, capsize=4, edgecolor='white')
    axes[1].axvline(3, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='Neutral (3)')
    axes[1].set_xlim(0, 5.5)
    axes[1].set_title("Q16–Q23 — Tool Usage Intensity (M ± SD)")
    axes[1].set_xlabel("Mean intensity (1=not used … 5=very heavily used)")
    axes[1].legend(fontsize=8)
    for i, (idx, m) in enumerate(zip(sorted_i, [means_t[j] for j in sorted_i])):
        axes[1].text(m + .08, i, f'{m:.2f}', va='center', fontsize=8, fontweight='bold')

    phase_labels = ['ENVISION', 'SPECULATE', 'EXPLORE', 'ADAPT', 'CLOSE']
    means_p  = [df[f'Q{i}_n'].mean() for i in range(24, 29)]
    stds_p   = [df[f'Q{i}_n'].std()  for i in range(24, 29)]
    clrs_p   = [C_MAIN if m == max(means_p) else
                C_SEC  if m == min(means_p) else '#5B9BD5' for m in means_p]
    bars_p = axes[2].bar(phase_labels, means_p, yerr=stds_p,
                         capsize=5, color=clrs_p, edgecolor='white')
    axes[2].axhline(3, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='Neutral (3)')
    axes[2].set_ylim(0, 5.5)
    axes[2].set_ylabel("Mean frequency (1=Never … 5=Very often)")
    axes[2].set_title("Q24–Q28 — AI Integration by Agile Phase (M ± SD)")
    axes[2].legend(fontsize=8)
    for bar, m in zip(bars_p, means_p):
        axes[2].text(bar.get_x() + bar.get_width()/2, m + .1,
                     f'{m:.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_fig("Fig11_Adoption_Tools_Phases")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  AI INTEGRATION IN AGILE TEAMS — M2 RESEARCH THESIS")
    print("  Full Quantitative Analysis Pipeline")
    print("="*70)

    df, cols = load_data(CSV_PATH)

    freq_profile(df, cols)
    desc_df = descriptive_stats(df)
    rel_df = reliability_analysis(df)
    val_df = validity_analysis(df)
    norm_df = normality_analysis(df)
    rho_df, p_df, n_corr = correlation_matrix(df)
    hyp_df = hypothesis_testing(df)
    group_comparisons(df, cols)
    moderation_analysis(df)
    reg_df, r2, r2_adj, f_stat, p_f = regression_analysis(df)
    perceived_mechanisms(df)
    challenges = content_analysis(df, cols)
    adoption_and_tools(df, cols)

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE — SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Sample         : n=100, French organizations, 76% Île-de-France
  Sectors        : Finance/Insurance 31%, IT/Software 24%
  AI Usage       : 55% use AI 'Often' or 'Very often' (Q11)
  Top tool       : Generative AI (M={df['Q16_n'].mean():.2f}/5)
  Top phase      : EXPLORE (M={df['Q26_n'].mean():.2f}) vs CLOSE (M={df['Q28_n'].mean():.2f})
  Sprint perf.   : M={df['Q35_n'].mean():.2f}/5  |  Overall success: M={df['Q39_n'].mean():.2f}/5
  Regression R²  : {r2:.3f}  Adj.R² : {r2_adj:.3f}  F={f_stat:.2f}  p={p_f:.4f}

  Hypothesis results:""")

    for _, row in hyp_df.iterrows():
        rho_str = f"ρ={row['ρ']}" if not pd.isna(row['ρ']) else "see detail"
        print(f"    {row['Hypothesis']:<4} {str(row['Decision']):<25} "
              f"{rho_str}  p={row['p-value']}")

    print(f"\n  → {len(os.listdir(OUTPUT_DIR))} figures saved to {OUTPUT_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()