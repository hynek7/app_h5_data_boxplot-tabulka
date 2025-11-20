import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import h5py
import pyarrow 
import os
import glob

# --- KONSTANTY A NASTAVENÍ CEST ---
# Zde si můžete nastavit absolutní cestu, pokud chcete obejít problém s pracovním adresářem
# Pokud soubory leží ve stejném adresáři jako skript, stačí ponechat relativní cestu:
H5_FILE_PATH = 'petacc3-student_project.h5'
ANNOT_FILE_PATH = 'petacc3-for_student_project-annot.feather'

# --- POMOCNÉ FUNKCE ---

@st.cache_data
def load_gene_annotation():
    """Načte anotace genů z .feather souboru."""
    try:
        annot_df = pd.read_feather(ANNOT_FILE_PATH)
        st.sidebar.success(f"Gene annotation loaded: {len(annot_df)} genes.")
        return annot_df
    except FileNotFoundError:
        st.error(f"Gene annotation file '{ANNOT_FILE_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading gene annotation: {e}")
        return None

@st.cache_data
def load_data():
    """Načte klinická data, genovou expresi a anotace."""
    
    clinical_df = None
    expression_data = None
    gene_names = None
    
    annot_df = load_gene_annotation()
    
    try:
        # 1. Načtení clinical_data.csv (primární)
        try:
            clinical_df = pd.read_csv('clinical_data.csv')
        except FileNotFoundError:
            clinical_df = None

        # 2. Načtení dat z HDF5 souboru
        try:
            with h5py.File(H5_FILE_PATH, 'r') as f:
                
                # --- Načtení Expresních dat (X) ---
                if 'X' in f:
                    x_group = f['X']
                    if all(key in x_group for key in ['axis0', 'axis1', 'block0_values']):
                        axis0_data = x_group['axis0'][:] # Geny (sloupce)
                        axis1_data = x_group['axis1'][:] # Vzorky (řádky/index)
                        matrix_data = x_group['block0_values'][:] # Data exprese
                        
                        # Dekódování na UTF-8
                        axis0_data = [name.decode('utf-8') if isinstance(name, bytes) else name for name in axis0_data]
                        axis1_data = [name.decode('utf-8') if isinstance(name, bytes) else name for name in axis1_data]
                        
                        if matrix_data.shape[0] == len(axis1_data) and matrix_data.shape[1] == len(axis0_data):
                            expression_data = pd.DataFrame(
                                data=matrix_data,
                                index=axis1_data, 
                                columns=axis0_data
                            )
                            gene_names = axis0_data
                        else:
                            st.warning("HDF5 Error: Expression matrix dimensions do not match axis labels.")
                    else:
                        st.warning("HDF5 Error: Missing required datasets (axis0, axis1, or block0_values) in 'X' group.")


                # --- Načtení klinických dat z HDF5 ---
                clinical_from_h5 = None
                if 'clinical' in f and 'axis1' in f['clinical'] and 'block0_values' in f['clinical']:
                    clinical_group = f['clinical']
                    if 'axis0' in clinical_group:
                        clinical_columns = [name.decode('utf-8') if isinstance(name, bytes) else name for name in clinical_group['axis1'][:]]
                        clinical_samples = [name.decode('utf-8') if isinstance(name, bytes) else name for name in clinical_group['axis0'][:]]
                        clinical_matrix = clinical_group['block0_values'][:]
                        
                        if clinical_matrix.shape[0] == len(clinical_samples) and clinical_matrix.shape[1] == len(clinical_columns):
                            clinical_from_h5 = pd.DataFrame(
                                clinical_matrix,
                                index=clinical_samples,
                                columns=clinical_columns
                            )
                        else:
                            st.warning("HDF5 Error: Clinical matrix dimensions do not match axis labels.")
                
                # Přepíše CSV, pokud jsou data z H5 validní
                if clinical_from_h5 is not None and len(clinical_from_h5) > 0:
                    clinical_df = clinical_from_h5
                    
        except FileNotFoundError:
            # Varování, které se zobrazovalo, pokud soubor nebyl nalezen
            st.warning(f"HDF5 file '{H5_FILE_PATH}' not found. Gene expression analysis is disabled.")
        except Exception as e:
            st.error(f"Error accessing HDF5 file: {e}")
        
        # Kontrola a čištění dat
        if clinical_df is None:
            st.error("Fatal: Clinical data not found. Stopping.")
            return None, None, None, None
            
        # *** FINÁLNÍ A ROBUSTNÍ OPRAVA INDEXU A SLOUČENÍ ***
        
        # Kontrola a čištění dat
        if clinical_df is None:
            st.error("Fatal: Clinical data not found. Stopping.")
            return None, None, None, None
            
        # *** FINÁLNÍ KONTROLA A PŘÍPRAVA DAT ***
        
        # 1. Hledání ID sloupce v Clinical Data
        id_cols = ['patient_id', 'SampleID', 'icgc_donor_id', 'sample_id', 'submitter_id']
        found_id_col = None
        for col_name in id_cols:
            if col_name in clinical_df.columns:
                found_id_col = col_name
                break
        
        # 2. Nastavení indexu v Clinical Data a expresních datech
        def clean_index_simple(df, id_col):
             # Nastaví index a vyčistí ho (string, bez mezer)
             if id_col and id_col in df.columns:
                 df = df.set_index(id_col, drop=True)
             
             # Standardizace indexu (stripping)
             df.index = df.index.astype(str).str.strip()
             
             return df
             
        # Čištění a nastavení indexu klinických dat (zachováváme všechny)
        clinical_df = clean_index_simple(clinical_df, found_id_col)
        
        # Čištění expresních dat
        if expression_data is not None:
             # Expresní data musí mít unikátní index pro reindex v main()
             # Agresivní čištění pouze pro Expression Data, aby se srovnala s Clinical ID
             def clean_index_aggressive(index):
                 index = index.astype(str).str.strip()
                 # Odstranění běžných prefixů/sufixů
                 index = index.str.replace(r'[PETACC3_\-\.]', '', regex=True)
                 index = index.str.replace(r'[\D_]', '', regex=True) 
                 try:
                    index = index.astype(int).astype(str)
                 except ValueError:
                    pass
                 return index

             expression_data.index = clean_index_aggressive(expression_data.index)
             
             # Odstranění duplicit VÝHRADNĚ z expression_data
             if expression_data.index.has_duplicates:
                 original_len = len(expression_data)
                 expression_data = expression_data[~expression_data.index.duplicated(keep='first')]
                 st.warning(f"Removed {original_len - len(expression_data)} duplicate patient IDs from expression data for alignment.")

        # Ostatní čištění
        clinical_df = clean_clinical_data(clinical_df)
        
        # Vracíme kompletní klinická data a vyčištěná expresní data
        return clinical_df, expression_data, gene_names, annot_df
        
    except Exception as e:
        st.error(f"Top-level error loading data: {e}")
        return None, None, None, None


def clean_clinical_data(df):
    """Čistí a normalizuje klinická data."""
    df_clean = df.copy()

    # Konverze časových a událostních sloupců
    bool_columns = ['os.event', 'rfs.event', 'pfs.event']
    for col in bool_columns:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0}).fillna(df_clean[col])
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)

    # Normalizace kategorií
    for col in ['stage', 'grade', 'site', 'trtgrp', 'MSI']:
        if col in df_clean.columns:
            # trim whitespace a konverze na string
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # převod římských číslic → arabské pro Stage
            if col == 'stage':
                roman_map = {"I": "1", "II": "2", "III": "3", "IV": "4"}
                df_clean[col] = df_clean[col].replace(roman_map)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')

    categorical_columns = ['trtgrp', 'grade', 'site', 'stage', 'BRAF', 'KRAS', 'MSI']
    for col in categorical_columns:
        if col in df_clean.columns:
            # Převedeme na kategorii (ignorujeme chyby, pokud už je částečně numerická)
            try:
                df_clean[col] = df_clean[col].astype('category')
            except:
                pass 
                
    return df_clean

def merge_gene_info(expression_data, annot_df):
    """
    Sloučí genové anotace s daty exprese s použitím normalizace Gene.Symbol.
    """
    if expression_data is None or annot_df is None:
        return None
        
    if 'Gene.Symbol' not in annot_df.columns:
        st.error("Annotation file is missing the 'Gene.Symbol' column.")
        return None

    # 1. Normalizace anotací pro klíč 'Gene.Symbol'
    annot_df = annot_df.copy()
    # Použijeme Gene.Symbol_Clean jako index pro spolehlivé vyhledávání
    annot_df['Gene.Symbol_Clean'] = annot_df['Gene.Symbol'].astype(str).str.strip()
    annot_indexed = annot_df.set_index('Gene.Symbol_Clean', drop=False) 

    # 2. Normalizace klíčů z expresní matice (což jsou Gene Symbols)
    exp_genes_clean = pd.Index([col.strip() for col in expression_data.columns])
    
    # 3. Sloučení: Najít společné geny
    common_genes = exp_genes_clean.intersection(annot_indexed.index)
    
    if len(common_genes) == 0:
        st.warning("Warning: No matching gene symbols found between expression data and annotation file. Gene info disabled.")
        return None

    # Vrátí DataFrame indexovaný Symboly genů z expresní matice
    merged_info = annot_indexed.loc[common_genes]
    
    return merged_info

# --- VIZUALIZAČNÍ FUNKCE (Zůstávají stejné) ---

def plot_survival_analysis(df, biomarker, survival_type='os'):
    """Vytváří Kaplan-Meierovu křivku přežití."""
    try:
        kmf = KaplanMeierFitter()
        fig = go.Figure()
        
        # Diskretizace pro numerické biomarkery (např. genová exprese)
        if pd.api.types.is_numeric_dtype(df[biomarker]) and df[biomarker].nunique() > 10:
            threshold = df[biomarker].median()
            df_analysis = df.copy()
            df_analysis[biomarker] = np.where(df_analysis[biomarker] > threshold, 'High', 'Low')
            unique_values = ['High', 'Low']
        else:
            df_analysis = df
            unique_values = df_analysis[biomarker].dropna().unique()

        valid_data = df_analysis[[biomarker, f'{survival_type}.time', f'{survival_type}.event']].dropna()
        
        if len(valid_data) == 0 or valid_data[f'{survival_type}.event'].sum() == 0:
            return go.Figure(), None, None
        
        unique_values = sorted(unique_values)
        groups_data = []
        small_at_risk = None
        
        for value in unique_values:
            mask = valid_data[biomarker] == value
            subset = valid_data[mask]
            
            if len(subset) > 0 and subset[f'{survival_type}.event'].sum() > 0:
                time_col = f'{survival_type}.time'
                event_col = f'{survival_type}.event'
                
                kmf.fit(subset[time_col], subset[event_col], label=f"{value}")
                
                event_table = kmf.event_table.reset_index()
                small_table_value = event_table[["event_at", "at_risk"]].copy()
                small_table_value.rename(columns={"at_risk": f"At risk ({value})"}, inplace=True)

                if small_at_risk is None:
                    small_at_risk = small_table_value
                else:
                    small_at_risk = small_at_risk.merge(small_table_value, on="event_at", how="outer")
                
                fig.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_.iloc[:, 0],
                    mode='lines',
                    name=f"{value} (n={len(subset)})",
                    line=dict(width=3),
                    hovertemplate='Time: %{x}<br>Survival: %{y:.3f}<extra></extra>'
                ))
                
                groups_data.append((subset[time_col], subset[event_col]))
        
        logrank_result = None
        if len(groups_data) >= 2:
            try:
                # Provede Log-rank test pouze pro první dvě skupiny
                result = logrank_test(groups_data[0][0], groups_data[1][0], 
                                     groups_data[0][1], groups_data[1][1])
                logrank_result = {
                    'p_value': result.p_value,
                    'test_statistic': result.test_statistic
                }
            except Exception:
                pass
        
        survival_title = 'Overall Survival' if survival_type == 'os' else 'Recurrence-Free Survival'
        title = f'{survival_title} by {biomarker}'
        if logrank_result:
            title += f' (Log-rank p={logrank_result["p_value"]:.4f})'
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (Months)',
            yaxis_title='Survival Probability',
            hovermode='x unified'
        )
        
        return fig, logrank_result, small_at_risk
        
    except Exception as e:
        st.error(f"Error creating survival plot: {e}")
        return go.Figure(), None, None


def plot_biomarker_distribution(df, biomarker):
    """Vykreslí distribuci klinického biomarkeru."""
    try:
        if pd.api.types.is_numeric_dtype(df[biomarker]) and df[biomarker].nunique() > 10:
            fig = px.histogram(df, x=biomarker, 
                             title=f"Distribution of {biomarker}",
                             marginal="box",
                             hover_data=df.columns)
        else:
            value_counts = df[biomarker].value_counts().reset_index()
            value_counts.columns = [biomarker, 'count']
            fig = px.bar(value_counts, x=biomarker, y='count',
                         title=f"Distribution of {biomarker}",
                         color=biomarker)
        
        return fig
    except Exception as e:
        st.error(f"Error creating distribution plot: {e}")
        return px.histogram()

def plot_gene_expression(df, biomarker):
    """Vykreslí distribuci genové exprese."""
    try:
        fig = px.histogram(df, x=biomarker, 
                           title=f"Distribution of {biomarker} Expression",
                           marginal="box",
                           labels={biomarker: 'Expression Level'})
        return fig
    except Exception as e:
        st.error(f"Error creating gene expression plot: {e}")
        return px.histogram()

def plot_treatment_response(df, biomarker):
    """Vykreslí průměrné přežití podle biomarkeru a léčebné skupiny."""
    try:
        response_data = df.groupby([biomarker, 'trtgrp']).agg({
            'os.time': ['mean', 'std', 'count'],
            'rfs.time': ['mean', 'std'],
            'os.event': 'mean'
        }).round(2).reset_index()
        response_data.columns = ['_'.join(col).strip('_') for col in response_data.columns.values]
        
        fig = px.bar(
            response_data, 
            x=biomarker, 
            y='os.time_mean', 
            color='trtgrp', 
            barmode='group',
            error_y='os.time_std',
            title=f"Average Overall Survival by {biomarker} and Treatment",
            labels={'os.time_mean': 'Mean Survival Time (Months)'},
            hover_data=['os.time_count']
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating treatment response plot: {e}")
        return px.bar()


def create_biomarker_selection_table(clinical_df):
    """Vytváří seznam klinických biomarkerů v postranním panelu."""
    st.sidebar.subheader("📋 Clinical Biomarkers")
    
    clinical_biomarkers = {
        'Stage': 'stage',
        'MSI Status': 'MSI',
        'Tumor Grade': 'grade',
        'Tumor Site': 'site',
        'Treatment Group': 'trtgrp',
        'Gender': 'gender',
        'Age Group': 'age_group'
    }
    
    available_biomarkers = {dname: cname for dname, cname in clinical_biomarkers.items() if cname in clinical_df.columns}
    
    selected_biomarker = None
    
    for display_name, col_name in available_biomarkers.items():
        raw_vals = clinical_df[col_name].astype(str).dropna().unique().tolist()
        unique_values = ['All'] + sorted([v for v in raw_vals if v != 'nan'])

        selected_value = st.sidebar.selectbox(
            display_name,
            options=unique_values,
            index=0,
            key=f"biomarker_{col_name}"
        )

        if selected_value != 'All':
            selected_biomarker = (display_name, col_name, selected_value)
            break
    
    return selected_biomarker


# --- HLAVNÍ FUNKCE APLIKACE ---

def main():
    st.set_page_config(page_title="Colorectal Cancer Biomarker Explorer", 
                        layout="wide", 
                        page_icon="🔬")
    
    st.title("🔬 Colorectal Cancer Biomarker Explorer")
    st.markdown("Explore predictive and prognostic biomarkers for treatment response and survival")
    
    clinical_df, expression_data, gene_names, annot_df = load_data()
    
    if clinical_df is None:
        st.stop()
        
    # Sloučení anotací genů
    merged_gene_info = None
    if expression_data is not None and annot_df is not None:
        merged_gene_info = merge_gene_info(expression_data, annot_df)
    
    st.sidebar.header("🔧 Analysis Parameters")
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Patients:** {len(clinical_df)}")
    if expression_data is not None:
        st.sidebar.write(f"**Genes (Expression):** {expression_data.shape[1]}")
    if merged_gene_info is not None:
        st.sidebar.write(f"**Genes (Annotated):** {len(merged_gene_info)}")
    
    # *** NOVÁ POZICE PRO VYKRESLENÍ WIDGETŮ FILTRŮ ***
    selected_gene_biomarker = None
    selected_gene_info = None
    
    # =======================================================
    # I. VÝBĚR GENU A ZOBRAZENÍ ANOTACE (PRVNÍ PRIORITA)
    # =======================================================
    if expression_data is not None:
        st.sidebar.markdown(
            """
            <div style="background-color: #cde2ee; padding: 10px; border-radius: 10px; border: 1px solid #0b5394;">
                <h3 style="color: #0b5394; margin: 0;">🧬 Select gene</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        selected_gene = st.sidebar.selectbox(
            "Select Gene",
            options=expression_data.columns.tolist(),
            index=0 if len(expression_data.columns) > 0 else 0,
            key="gene_selection_box",
            label_visibility="hidden"
        )
        
        # --- ZOBRAZENÍ ANOTACE ---
        if selected_gene:
            cleaned_selected_gene = selected_gene.strip()
            
            if merged_gene_info is not None and cleaned_selected_gene in merged_gene_info.index:
                selected_gene_info = merged_gene_info.loc[cleaned_selected_gene]
                # Nastavení biomakera pro analýzu
                selected_gene_biomarker = (f"Gene: {selected_gene}", selected_gene, "Expression")

                # Zobrazení informací o genu
                st.sidebar.markdown("---")
                # Používáme Gene.Symbol z anotace (který by měl být vyčištěný)
                st.sidebar.subheader(f"ℹ️ Gene Information: **{selected_gene_info['Gene.Symbol']}**")
                
                # Popis
                description = selected_gene_info['Gene.Description']
                st.sidebar.markdown(f"**Description:** {description}")
                
                # NCBI Link
                entrez_id = selected_gene_info['Entrez.Gene']
                if pd.notna(entrez_id) and entrez_id != '':
                    try:
                        entrez_id_int = int(entrez_id)
                        ncbi_link = f"https://www.ncbi.nlm.nih.gov/gene/{entrez_id_int}"
                        st.sidebar.markdown(f"[More info on NCBI Gene]({ncbi_link})")
                    except ValueError:
                         st.sidebar.markdown(f"*(Entrez ID: {entrez_id})*")
                
                st.sidebar.markdown("---")
            elif merged_gene_info is not None and cleaned_selected_gene not in merged_gene_info.index:
                st.sidebar.warning(f"Annotation not found for gene: {selected_gene}.")
    
    # =======================================================
    # II. VÝBĚR KLINICKÉHO BIOMARKERU
    # =======================================================

    # ----------------------------------------------------
    # Určení finálního biomakera pro analýzu a filtrování dat
    # ----------------------------------------------------
    
    biomarker_column = None
    biomarker_display = None
    
    # 1. Start s kompletní klinickou tabulkou
    analysis_df = clinical_df.copy()
    
    # *** KLÍČOVÝ KROK: VYKRESLENÍ FILTRŮ A ZÍSKÁNÍ HODNOT (VOLÁNO POUZE ZDE!) ***
    # Tato funkce VYKRESLÍ widgety a vrátí vybranou hodnotu.
    clinical_filter = create_biomarker_selection_table(clinical_df) 
    
    # 2. Aplikujeme KLINICKÝ FILTR, pokud byl vybrán
    if clinical_filter:
        filter_display, filter_column, filter_value = clinical_filter
        
        if filter_value != 'All':
            st.sidebar.info(f"Filtrování dat pro: **{filter_display} = {filter_value}**")

            # Filtrace
            filtered = analysis_df[analysis_df[filter_column].astype(str).str.strip() == str(filter_value).strip()]
            
            if len(filtered) == 0:
                st.warning(f"No patients match {filter_display} = {filter_value}. Showing all patients instead.")
            else:
                analysis_df = filtered # Nyní je analysis_df filtrovaná!
    
    # 3. Určení primárního biomarkera pro analýzu (Gen nebo Klinický)
    
    if selected_gene_biomarker:
        # Gen má prioritu.
        biomarker_display, biomarker_column, _ = selected_gene_biomarker
        
        # Přidání dat exprese k analysis_df
        if expression_data is not None:
             # Zarovnání na index Patient ID
             gene_data_aligned = expression_data[biomarker_column].reindex(analysis_df.index)
             analysis_df[biomarker_column] = gene_data_aligned
             
    elif clinical_filter:
        # Pokud byl klinický filtr použit, použijeme ho jako biomarker
        biomarker_display, biomarker_column, _ = clinical_filter
        
    else:
        # 4. Defaultní biomarker (pokud není vybráno nic)
        clinical_biomarkers = {'Stage': 'stage', 'MSI Status': 'MSI'}
        available_biomarkers = {d: c for d, c in clinical_biomarkers.items() if c in clinical_df.columns}
        if available_biomarkers:
            biomarker_display = list(available_biomarkers.keys())[0]
            biomarker_column = available_biomarkers[biomarker_display]
        else:
            st.error("No suitable biomarker columns found for default analysis.")
            st.stop()
            
    # *** Přenese finalizovaná (filtrovaná + genová) data do clinical_df ***
    clinical_df = analysis_df
            
    # ----------------------------------------------------
    # Vykreslení
    # ----------------------------------------------------
    
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Survival Analysis", "Biomarker Distribution", "Treatment Response"]
    )

    survival_code = 'os'
    if analysis_type == "Survival Analysis":
        survival_type = st.sidebar.radio(
            "Survival Type",
            ["Overall Survival", "Recurrence-Free Survival"]
        )
        survival_code = {'Overall Survival': 'os', 'Recurrence-Free Survival': 'rfs'}[survival_type]
 
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {analysis_type} for {biomarker_display}")
        
        if analysis_type == "Survival Analysis":
            fig, logrank_result, small_at_risk = plot_survival_analysis(clinical_df, biomarker_column, survival_code)
            st.plotly_chart(fig, use_container_width=True)

            if small_at_risk is not None:
                st.subheader("📋 Patients at Risk")
                small_clean = small_at_risk.copy().fillna(method='ffill')
                # Zobrazení pouze sloupce At risk pro první skupinu (simplifikace)
                st.dataframe(small_clean.iloc[:, [0, 2]].rename(columns={"event_at": "Month"}).dropna()) 
                
            if logrank_result:
                st.info(f"**Log-rank Test:** p-value = {logrank_result['p_value']:.4f}")
                
        elif analysis_type == "Biomarker Distribution":
            if selected_gene_biomarker:
                fig = plot_gene_expression(clinical_df, biomarker_column)
            else:
                fig = plot_biomarker_distribution(clinical_df, biomarker_column)
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Treatment Response":
            if 'trtgrp' in clinical_df.columns and biomarker_column != 'trtgrp':
                fig = plot_treatment_response(clinical_df, biomarker_column)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Treatment Response plot requires 'trtgrp' column and a different grouping biomarker.")
    
    with col2:
        st.subheader(f"📈 Summary Statistics for **{biomarker_display}**")
        
        if biomarker_column in clinical_df.columns:
            
            if pd.api.types.is_numeric_dtype(clinical_df[biomarker_column]):
                
                st.dataframe(clinical_df[biomarker_column].describe().to_frame().T.round(2))
                
                fig_box = px.box(
                    clinical_df,
                    y=biomarker_column,
                    points="all",
                    title=f"{biomarker_display} – Boxplot"
                )
                st.plotly_chart(fig_box, use_container_width=True)

            else:
                st.write(f"**Value Counts:**")
                st.dataframe(clinical_df[biomarker_column].value_counts().to_frame())

                st.write(f"**Overall Survival time by {biomarker_display}:**")

                if "os.time" in clinical_df.columns:
                    fig_box = px.box(
                        clinical_df,
                        x=biomarker_column,
                        y="os.time",
                        points="all",
                        title=f"OS Time by {biomarker_display}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("OS time column not available in data.")
    

if __name__ == "__main__":
    main()