import streamlit as st
import sys
import os

# --- New Debugging Block ---
# Get the full path to your home directory and the project folder
adapter_path = os.path.join(os.path.dirname(__file__), "Retrieval-Augmented-Time-Series-Forecasting")
full_file_path = os.path.join(adapter_path, 'chronos_bolt_adapter.py')


# Check if the file actually exists
if os.path.exists(full_file_path):
    st.success("Adapter file found! Adding path to sys.path.")
    # Add this path to the list of places Python looks for modules
    if adapter_path not in sys.path:
        sys.path.insert(0, adapter_path)
else:
    st.error(f"File NOT found at the path above. Please check your directory and file name.")
    st.stop() # Stop the app if the file isn't found
# --- End of New Block ---



import pandas as pd
import numpy as np
import torch
import time
import altair as alt
from tqdm.auto import tqdm
import logging
from pathlib import Path

# --- User's Imports ---
# Make sure you have this file in the same directory
try:
    from chronos_bolt_adapter import ChronosBoltAdapter
except ImportError:
    st.error("Error: `chronos_bolt_adapter.py` not found. Please place it in the same directory as this app.")
    st.stop()

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RetrievalAugApp")

# ----------------------------------------------------------------------
# CONFIG (from user script, now as constants)
# ----------------------------------------------------------------------
PREDICTION_LENGTH = 7
HISTORY_LENGTH = 100
CHRONOS_MODEL_ID = "amazon/chronos-bolt-small"
# This is used to define the *latest* data used to build the candidate bank
# Any data after this is reserved for "live" evaluation
EVAL_START_FOR_BANK = pd.Timestamp("2024-05-15") 
OVERLAP_THRESH = 0.70  # >= 70% overlap -> drop duplicate window
AUG_LEN = 2 * HISTORY_LENGTH + PREDICTION_LENGTH

# !!!!!!!!!!!!!!!!!! CRUCIAL !!!!!!!!!!!!!!!!!!
# !!! UPDATE THIS PATH TO YOUR CSV FILE !!!
DATA_FILEPATH = "data file path"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ----------------------------------------------------------------------
# Helper Function (from user script)
# ----------------------------------------------------------------------
def build_date_index(dates_map):
    out = {}
    for uid, dl in dates_map.items():
        dlist = [pd.Timestamp(d) for d in dl]
        if not pd.Index(dlist).is_monotonic_increasing:
            order = np.argsort(dlist)
            dlist = [dlist[i] for i in order]
        idx_map = {d: i for i, d in enumerate(dlist)}
        out[uid] = (dlist, idx_map)
    return out

def window_overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    length = max(a_end - a_start, 1)
    return inter / length

# ----------------------------------------------------------------------
# Data and Model Loading (Cached)
# ----------------------------------------------------------------------

@st.cache_data
def load_data(filepath):
    """Loads and preprocesses data into series_map and dates_map."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at '{filepath}'. Please update `DATA_FILEPATH` in the script.")
        return None, None, None, None

    data['date'] = pd.to_datetime(data['date'])
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    long_df = pd.melt(df, id_vars=['date'],
                      value_vars=[col for col in df.columns if col != 'date'],
                      var_name='unique_id',
                      value_name='y')
    
    long_df.rename(columns={'date': 'ds'}, inplace=True)
    long_df = long_df[['unique_id', 'ds', 'y']]
    
    Y_df = long_df.rename(columns={'y': 'target'})
    
    series_map = {}
    dates_map = {}
    for uid, grp in Y_df.groupby('unique_id'):
        # Ensure data is sorted by date for correct indexing
        grp = grp.sort_values('ds')
        vals = grp['target'].to_numpy(dtype=np.float32)
        ds_list = grp['ds'].tolist()
        series_map[uid] = vals
        dates_map[uid] = ds_list
        
    date_index = build_date_index(dates_map)
    station_uids = sorted(list(series_map.keys()))
    
    logger.info(f"Loaded {len(series_map)} series.")
    return series_map, dates_map, date_index, station_uids

@st.cache_data
def build_candidate_bank(_series_map, _dates_map, H, P, eval_start):
    """Builds the retrieval candidate bank. Cached for performance."""
    cand_full_segments = []
    cand_series = []
    cand_end_idx = []
    cand_start_idx = []
    
    t_build = time.time()
    for uid, values in _series_map.items():
        x = np.asarray(values, dtype=float)
        if len(x) < H + P:
            continue
            
        dates_uid = [pd.Timestamp(d) for d in _dates_map[uid]]
        stride = 1
        for end in range(H, len(x) - P + 1, stride):
            end_date = dates_uid[end - 1]
            if end_date >= eval_start:
                continue
                
            start = end - H
            hist_seg = x[start:end]
            fut_seg = x[end:end+P]

            if np.isnan(hist_seg).all() or np.isnan(fut_seg).all():
                continue

            hist_seg = pd.Series(hist_seg).ffill().bfill().values
            fut_seg = pd.Series(fut_seg).ffill().bfill().values
            
            full_seg = np.concatenate([hist_seg, fut_seg])
            cand_full_segments.append(full_seg.astype(np.float32))
            cand_series.append(uid)
            cand_end_idx.append(end)
            cand_start_idx.append(start)

    cand_full_segments = (np.stack(cand_full_segments)
                          if cand_full_segments else np.zeros((0, H+P), dtype=np.float32))
    cand_series = np.array(cand_series)
    cand_end_idx = np.array(cand_end_idx)
    cand_start_idx = np.array(cand_start_idx)
    
    logger.info(f"[BUILD] Candidate segments: {cand_full_segments.shape} "
                f"built in {time.time() - t_build:.2f}s")
                
    uid_to_cand_indices = {}
    for i, s in enumerate(cand_series):
        uid_to_cand_indices.setdefault(s, []).append(i)
        
    cand_hist_tensor = torch.from_numpy(cand_full_segments[:, :H])
    
    bank = {
        "segments": cand_full_segments,
        "series": cand_series,
        "end_idx": cand_end_idx,
        "start_idx": cand_start_idx,
        "hist_tensor": cand_hist_tensor,
        "uid_to_indices": uid_to_cand_indices
    }
    return bank

@st.cache_resource
def load_model(model_id):
    """Loads the Chronos model. Cached as a resource."""
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                   else torch.float32)

    logger.info(f"Loading Chronos model '{model_id}' on {device_map}...")
    chronos = ChronosBoltAdapter(
        model_id=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        faux_num_samples=20 # From user's script
    )
    logger.info("Model loaded successfully.")
    return chronos

# ----------------------------------------------------------------------
# Core Logic Functions (Refactored from user script)
# ----------------------------------------------------------------------

def retrieve_topn_simraf(query_series, query_uid, cutoff_idx, top_n, bank, H, P, overlap_thresh):
    """Refactored retrieval function."""
    if top_n == 0:
        return np.array([], dtype=int), np.array([], dtype=float), query_series[cutoff_idx - H: cutoff_idx]

    if cutoff_idx is None or cutoff_idx < H or cutoff_idx + P > len(query_series):
        return None

    ctx = query_series[cutoff_idx - H: cutoff_idx]
    ctx = pd.Series(ctx).ffill().bfill().values
    
    q = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        dists = torch.cdist(q, bank["hist_tensor"], p=2).squeeze(0)

    # Invalidate segments from the same series that are too close
    for idx in bank["uid_to_indices"].get(query_uid, []):
        if bank["end_idx"][idx] >= (cutoff_idx - 1): # Overlaps or is in the future
            dists[idx] = float("inf")

    valid = (~torch.isinf(dists)).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        logger.warning(f"No valid retrieval candidates found for {query_uid} at cutoff {cutoff_idx}")
        return None # No valid candidates found

    d_valid = dists[valid]
    order = torch.argsort(d_valid, descending=False)

    chosen, d_sel = [], []
    for pos in order:
        if len(chosen) >= top_n:
            break
        ci = int(valid[pos])
        keep = True
        # Overlap filter
        for t in chosen:
            if bank["series"][ci] == bank["series"][t]:
                ov = window_overlap_ratio(
                    bank["start_idx"][ci], bank["end_idx"][ci],
                    bank["start_idx"][t], bank["end_idx"][t]
                )
                if ov >= overlap_thresh:
                    keep = False
                    break
        if keep:
            chosen.append(ci)
            d_sel.append(float(d_valid[pos]))
            
    return np.array(chosen, dtype=int), np.array(d_sel, dtype=float), ctx

def build_augmented_variants(query_ctx, top_indices, bank, H, P, aug_len):
    """Refactored variant builder."""
    ctx_t = torch.tensor(query_ctx, dtype=torch.float32)
    m_c = float(ctx_t.mean())
    s_c = float(ctx_t.std(unbiased=False) + 1e-7)
    ctx_n = (ctx_t - m_c) / s_c
    variants, stats = [], []
    
    # Baseline variant (with zero-padding for augmentation format)
    zero_prefix = torch.zeros(H + P, dtype=torch.float32)
    variants.append(torch.cat([zero_prefix, ctx_n]))
    stats.append((m_c, s_c))

    # Augmented variants
    for ci in top_indices:
        seg_full = bank["segments"][ci]
        seg_t = torch.tensor(seg_full, dtype=torch.float32)
        m_s = float(seg_t.mean())
        s_s = float(seg_t.std(unbiased=False) + 1e-7)
        seg_n = (seg_t - m_s) / s_s
        
        # Level shift (from user's code)
        shift = ctx_n[0] - seg_n[-1] # This seems to align end of hist with start of query?
                                     # User code had seg_n[-1] but maybe meant seg_n[H-1]?
                                     # Using user's original logic: seg_n[-1]
        

        
        shift = ctx_n[0] - seg_n[-1]
        seg_n = seg_n + shift
        
        variants.append(torch.cat([seg_n, ctx_n]))
        stats.append((m_c, s_c)) # Note: denormalizing with query stats

    for v in variants:
        assert v.shape[0] == aug_len, f"Variant length mismatch {v.shape[0]} != {aug_len}"
        
    return variants, stats

def run_forecast(chronos, variants, stats, top_n, top_dists, P, aggregate_mode="mean"):
    """Runs the model and aggregates predictions."""
    with torch.no_grad():
        samples = chronos.predict(variants, prediction_length=P, num_samples=1)
        
    fc_matrix = samples[:, 0, :].cpu().numpy()
    
    fc_matrix_denorm = np.array([
        fc_matrix[i] * stats[i][1] + stats[i][0]
        for i in range(len(fc_matrix))
    ])
    
    baseline_fc = fc_matrix_denorm[0] # First variant is always baseline
    
    if top_n == 0:
        final_fc = baseline_fc
    else:
        # Aggregate [baseline, aug1, aug2, ...]
        if aggregate_mode == "mean":
            final_fc = fc_matrix_denorm.mean(axis=0)
        else: # Weighted (from user script)
            alpha = 1.0 # User's script didn't define alpha, setting a default
            anchor = np.median(top_dists) if top_dists.size > 0 else 0
            d_all = np.concatenate([[anchor], top_dists]) # Dists for [baseline_anchor, aug1, ...]
            w = np.exp(-alpha * d_all); w /= w.sum()
            final_fc = (fc_matrix_denorm * w[:,None]).sum(axis=0)
            
    return final_fc, baseline_fc, fc_matrix_denorm

############# MIRAF ###############

# --- MIRAF retrieval (Mutual Information over HISTORY only) ---
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd

def retrieve_topn_miraf(
    query_series,
    query_uid,
    cutoff_idx,
    top_n,
    bank,
    H,
    P,
    overlap_thresh,
    n_neighbors=3,
    var_eps=1e-8,
):
    """
    Returns:
      top_idx   : np.ndarray[int]   (<= top_n)
      top_scores: np.ndarray[float] (MI scores; higher is better)
      ctx       : np.ndarray[float] (H,)
    or None if retrieval cannot be done.
    """
    # Baseline mode (Top-N = 0) — same contract as SIM-RAF
    if top_n == 0:
        return np.array([], dtype=int), np.array([], dtype=float), query_series[cutoff_idx - H: cutoff_idx]

    # Bounds/guards — identical logic to SIM-RAF
    if cutoff_idx is None or cutoff_idx < H or cutoff_idx + P > len(query_series):
        return None

    # Build & clean query context (H points, ffill/bfill if needed)
    ctx = query_series[cutoff_idx - H: cutoff_idx]
    if len(ctx) != H or np.isnan(ctx).all():
        return None
    if np.isnan(ctx).any():
        ctx = pd.Series(ctx).ffill().bfill().values
    ctx = ctx.astype(float)

    # Candidates: use the history part only for MI
    segs = bank["segments"]
    if segs.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float), ctx
    cand_hist = segs[:, :H].astype(float)   # shape (N, H)

    # Prepare X (H, N) and y (H,) for MI
    X = cand_hist.T
    y = ctx

    # Drop constant-variance features to avoid MI issues
    var_mask = X.var(axis=0) > var_eps
    if not var_mask.any():
        return np.array([], dtype=int), np.array([], dtype=float), ctx

    X_use = X[:, var_mask]
    scores_sub = mutual_info_regression(
        X_use, y,
        discrete_features=False,
        n_neighbors=n_neighbors
    )

    # Rebuild full scores aligned to all candidates; invalid => -inf
    scores = np.full(X.shape[1], -np.inf, dtype=float)
    scores[var_mask] = scores_sub

    # Leakage avoidance: invalidate same-UID windows that touch/overlap future
    for idx in bank["uid_to_indices"].get(query_uid, []):
        if bank["end_idx"][idx] >= (cutoff_idx - 1):
            scores[idx] = -np.inf

    # If nothing valid, bail
    valid = ~np.isneginf(scores)
    if not valid.any():
        return None

    vidx = np.where(valid)[0]
    vscores = scores[valid]

    # Rank by MI (desc)
    order = np.argsort(vscores)[::-1]

    chosen, chosen_scores = [], []
    for pos in order:
        if len(chosen) >= top_n:
            break
        ci = int(vidx[pos])

        # Overlap suppression (≥ overlap_thresh) — only compare within same UID
        keep = True
        for t in chosen:
            if bank["series"][ci] == bank["series"][t]:
                a_start, a_end = bank["start_idx"][ci], bank["end_idx"][ci]
                b_start, b_end = bank["start_idx"][t],  bank["end_idx"][t]
                inter = max(0, min(a_end, b_end) - max(a_start, b_start))
                length = max(a_end - a_start, 1)
                overlap = inter / length
                if overlap >= overlap_thresh:
                    keep = False
                    break

        if keep:
            chosen.append(ci)
            chosen_scores.append(float(vscores[pos]))

    return np.array(chosen, dtype=int), np.array(chosen_scores, dtype=float), ctx


# ----------------------------------------------------------------------
# Streamlit App UI
# ----------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Retrieval-Augmented Forecasting for the Everglades 🌊")

# --- 1. Load Data and Models ---
# This runs once and is cached
with st.spinner("Loading data and building retrieval bank..."):
    series_map, dates_map, date_index, station_uids = load_data(DATA_FILEPATH)

if series_map is None:
    st.error("Failed to load data. Please check the `DATA_FILEPATH` constant.")
    st.stop()

with st.spinner("Loading Chronos forecasting model..."):
    bank = build_candidate_bank(series_map, dates_map, HISTORY_LENGTH, PREDICTION_LENGTH, EVAL_START_FOR_BANK)
    chronos = load_model(CHRONOS_MODEL_ID)

st.success("Data and model loaded successfully!")

# --- 2. Sidebar for User Inputs ---
st.sidebar.header("Forecast Configuration")

aug_method = st.sidebar.radio(
    "Select Augmentation Method:",
    ("Similarity (SimRAF)", "Mutual Information (MIRAF)")
)

selected_station = st.sidebar.selectbox(
    "Select a Station (unique_id):",
    station_uids,
    index=station_uids.index("NP205_stage") if "NP205_stage" in station_uids else 0
)

# Get valid dates for the selected station
station_dates = [pd.Timestamp(d) for d in dates_map[selected_station]]
# Must have enough history AND be after the candidate bank's last date
min_date = max(
    station_dates[HISTORY_LENGTH], 
    EVAL_START_FOR_BANK
)
# Must have room to predict
max_date = station_dates[-PREDICTION_LENGTH]

selected_date = st.sidebar.date_input(
    f"Select Forecast Date (Cutoff):",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    help=f"Select the last day of known history. The forecast will begin the next day. "
         f"Valid dates are between {min_date.date()} and {max_date.date()}."
)
selected_date = pd.Timestamp(selected_date)

#top_n = st.sidebar.slider(
#    "Number of Retrieved Samples (Top-N):",
#    min_value=0, max_value=10, value=4,
#    help="0 = Run in baseline mode (no retrieval)"
#)
top_n = 4

# --- 3. Run Forecast on Button Click ---
if st.button("Generate Forecast"):
    
    # --- Get Data & Cutoff ---
    series_vals = np.asarray(series_map[selected_station], dtype=float)
    
    def date_to_cutoff(uid, ds, date_index_map):
        ds_ts = pd.Timestamp(ds)
        if uid not in date_index_map: return None
        _, im = date_index_map[uid]
        return im.get(ds_ts, None)
        
    cutoff = date_to_cutoff(selected_station, selected_date, date_index)
    
    if cutoff is None:
        st.error(f"Selected date {selected_date.date()} not found in the index for station {selected_station}.")
        st.stop()

    st.header(f"Forecast for: {selected_station}")
    st.subheader(f"History up to: {selected_date.date()} | Method: {aug_method}")

    # --- 1. Retrieval ---
    with st.spinner("Retrieving similar segments..."):
        if "SimRAF" in aug_method:
            ret = retrieve_topn_simraf(
                series_vals, selected_station, cutoff, top_n,
                bank, HISTORY_LENGTH, PREDICTION_LENGTH, OVERLAP_THRESH
            )
            score_label = "Distance (L2)"            
        else:
            ret = retrieve_topn_miraf(
                series_vals, selected_station, cutoff, top_n,
                bank, HISTORY_LENGTH, PREDICTION_LENGTH, OVERLAP_THRESH,
                n_neighbors=3
            )
            score_label = "Mutual Information"

        if ret is None:
            st.error("Could not retrieve samples for this date (e.g., not enough history or no candidates found).")
            st.stop()
            
        top_idx, top_dists, query_ctx = ret

    # --- 2. Build Variants & Forecast ---
    with st.spinner("Building augmented contexts and running Chronos model..."):
        variants, stats = build_augmented_variants(
            query_ctx, top_idx, bank, HISTORY_LENGTH, PREDICTION_LENGTH, AUG_LEN
        )
        
        final_fc, baseline_fc, all_fcs = run_forecast(
            chronos, variants, stats, top_n, top_dists, PREDICTION_LENGTH, aggregate_mode="mean"
        )
    
    st.success("Forecast complete!")

    # --- 3. Display Results ---
    
    # --- Forecast Plot ---
    st.subheader("Forecast vs. Actual")
    
    # Get truth data
    truth_future = series_vals[cutoff:cutoff + PREDICTION_LENGTH]
    
    # Create a DataFrame for plotting
    forecast_dates = pd.date_range(start=selected_date, periods=PREDICTION_LENGTH + 1, freq="D")[1:]
    history_dates = pd.date_range(end=selected_date, periods=HISTORY_LENGTH, freq="D")

    df_hist = pd.DataFrame({"Date": history_dates, "Value": query_ctx, "Type": "History"})
    df_fc = pd.DataFrame({"Date": forecast_dates, "Value": final_fc, "Type": "Forecast (Augmented)"})
    df_truth = pd.DataFrame({"Date": forecast_dates, "Value": truth_future, "Type": "Actual"})
    df_base = pd.DataFrame({"Date": forecast_dates, "Value": baseline_fc, "Type": "Forecast (Baseline)"})

    # Combine all for plotting
    plot_df = pd.concat([df_hist, df_fc, df_truth, df_base], ignore_index=True)
    
    chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X("Date", title="Date"),
        y=alt.Y("Value", title="Water Level/Stage"),
        color=alt.Color("Type", title="Data Type"),
        tooltip=["Date", "Type", "Value"]
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # --- Retrieval Info ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Details")
        if top_n > 0 and len(top_idx) > 0:
            retrieved_info = []
            for i, idx in enumerate(top_idx):
                retrieved_info.append({
                    "Rank": i + 1,
                    "Series": bank["series"][idx],
                    "Start Date": dates_map[bank["series"][idx]][bank["start_idx"][idx]].date(),
                    "End Date (History)": dates_map[bank["series"][idx]][bank["end_idx"][idx]-1].date(),
                    score_label: f"{top_dists[i]:.4f}"
                })
            st.dataframe(pd.DataFrame(retrieved_info))
        elif top_n > 0:
            st.write("No segments were retrieved (Top-N > 0, but 0 returned).")
        else:
            st.write("Running in Baseline Mode (Top-N = 0).")

    # --- Augmentation Plot ---
    with col2:
        st.subheader("Retrieved Segments")
        if top_n > 0 and len(top_idx) > 0:
            # Base chart for query
            query_plot_df = pd.DataFrame({
                "Timestep": range(-HISTORY_LENGTH, 0), 
                "Value": query_ctx, 
                "Segment": "Query (History)"
            })
            
            retrieved_plots_df = [query_plot_df]
            for i, idx in enumerate(top_idx):
                segment_data = bank["segments"][idx]
                retrieved_plots_df.append(
                    pd.DataFrame({
                        "Timestep": range(-HISTORY_LENGTH, PREDICTION_LENGTH), # H+P
                        "Value": segment_data,
                        "Segment": f"Retrieved {i+1} ({bank['series'][idx]})"
                    })
                )
            
            aug_chart_df = pd.concat(retrieved_plots_df)
            
            aug_chart = alt.Chart(aug_chart_df).mark_line().encode(
                x=alt.X("Timestep", title="Timestep (0 = Forecast Start)"),
                y=alt.Y("Value", title="Water Level/Stage"),
                color=alt.Color("Segment", title="Segment"),
                tooltip=["Timestep", "Segment", "Value"]
            ).interactive()
            
            st.altair_chart(aug_chart, use_container_width=True)
            st.caption("Shows the H+P length retrieved segments vs. the H-length query.")
        else:
            st.write("No segments to visualize.")
