import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import time
from collections import OrderedDict

from src.ocr.pipeline import GlobalPipeline
from src.ocr.params import craftParams, craftComponentsParams, imageComponentsParams

# ==================== CONFIG ====================
st.set_page_config(layout="wide", page_title="Text Detection Pipeline", page_icon="📝")

# Load defaults from dataclasses
craft_defaults = craftParams()
craft_comp_defaults = craftComponentsParams()
img_comp_defaults = imageComponentsParams()

PRESETS = {
    "Custom": {
        "mag": float(craft_defaults.mag_ratio),
        "canvas": int(craft_defaults.canvas_size),
        "thresh": float(craft_comp_defaults.text_threshold),
        "connectivity": int(craft_comp_defaults.connectivity),
        "area": int(craft_comp_defaults.min_area if craft_comp_defaults.min_area else 10),
        "min_asp": float(craft_comp_defaults.min_aspect_ratio if craft_comp_defaults.min_aspect_ratio else 0.5),
        "max_asp": float(craft_comp_defaults.max_aspect_ratio if craft_comp_defaults.max_aspect_ratio else 2.0),
        "dist": float(craft_comp_defaults.min_dist if craft_comp_defaults.min_dist else 8.0),
        "threshold_method": str(img_comp_defaults.threshold),
        "img_aspect": float(img_comp_defaults.min_image_component_aspect_ratio),
        "axis_length": float(img_comp_defaults.min_image_component_axis_major_length_criterion),
        "similarity": float(img_comp_defaults.similarity_threshold),
        "min_box": tuple(img_comp_defaults.min_box_size),
        "max_box": tuple(img_comp_defaults.max_box_size),
        "char_aspect": float(img_comp_defaults.max_aspect_ratio),
        "filled_area": float(img_comp_defaults.max_filled_area_portion)
    },
    "High Precision": {
        "mag": 5.0, "canvas": 1280, "thresh": 0.8, "connectivity": 4,
        "area": 5, "min_asp": 0.3, "max_asp": 3.0, "dist": 8.0,
        "threshold_method": 'otsu', "img_aspect": 10.0, "axis_length": 30.0,
        "similarity": -15.0, "min_box": (30, 30), "max_box": (250, 250),
        "char_aspect": 1.5, "filled_area": 0.7
    },
    "Fast Processing": {
        "mag": 3.0, "canvas": 1280, "thresh": 0.5, "connectivity": 4,
        "area": 15, "min_asp": 0.5, "max_asp": 2.5, "dist": 8.0,
        "threshold_method": 'otsu', "img_aspect": 10.0, "axis_length": 30.0,
        "similarity": -15.0, "min_box": (30, 30), "max_box": (250, 250),
        "char_aspect": 1.5, "filled_area": 0.7
    },
    "Handwriting": {
        "mag": 6.0, "canvas": 1280, "thresh": 0.65, "connectivity": 4,
        "area": 8, "min_asp": 0.4, "max_asp": 4.0, "dist": 8.0,
        "threshold_method": 'otsu', "img_aspect": 10.0, "axis_length": 30.0,
        "similarity": -15.0, "min_box": (30, 30), "max_box": (250, 250),
        "char_aspect": 1.5, "filled_area": 0.7
    },
    "Dense Text": {
        "mag": 4.0, "canvas": 1280, "thresh": 0.7, "connectivity": 4,
        "area": 5, "min_asp": 0.4, "max_asp": 2.0, "dist": 8.0,
        "threshold_method": 'otsu', "img_aspect": 10.0, "axis_length": 30.0,
        "similarity": -15.0, "min_box": (30, 30), "max_box": (250, 250),
        "char_aspect": 1.5, "filled_area": 0.7
    }
}

# Session state
for key in ['pipeline_output', 'logged_figures', 'intermediate_logs']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'pipeline_output' else []
for key in ['pipeline_run_count', 'processing_time']:
    if key not in st.session_state:
        st.session_state[key] = 0

# Styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPERS ====================
@st.cache_resource
def load_pipeline(_craft_p, _comp_p, _img_p, dev):
    return GlobalPipeline(_craft_p, _comp_p, _img_p, dev)

def get_component_stats(labels, comp_id):
    """Calculate stats for single component"""
    mask = (labels == comp_id)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    
    return {
        'ID': comp_id, 
        'Pixels': int(np.sum(mask)),
        'Width': int(cols.max() - cols.min() + 1),
        'Height': int(rows.max() - rows.min() + 1),
        'X': int(cols.min()), 
        'Y': int(rows.min())
    }

def get_all_component_stats(labels, n_labels):
    """Get stats for all components"""
    return [s for i in range(1, n_labels) if (s := get_component_stats(labels, i)) is not None]

def resize_to_match(img, target_shape, is_label_map=False):
    """Resize image to target shape"""
    if isinstance(img, np.ndarray):
        if img.shape[:2] == target_shape[:2]:
            return img
        resample = Image.Resampling.NEAREST if is_label_map else Image.Resampling.LANCZOS
        return np.array(Image.fromarray(img).resize((target_shape[1], target_shape[0]), resample))
    return img

def get_all_pipeline_outputs(output):
    """Get all pipeline outputs as dict"""
    outputs = OrderedDict()
    outputs["Original Image"] = ("image", output.img_pil)
    outputs["Binary Image"] = ("gray", output.binary_img)
    
    if hasattr(output, 'score_text'):
        score = output.score_text.cpu().numpy() if hasattr(output.score_text, 'cpu') else output.score_text
        outputs["CRAFT Score Map"] = ("heatmap", score.squeeze())
    
    comp_attrs = [
        ('score_text_components', 'Initial Text Components'),
        ('filtered_text_components', 'Filtered Text Components'),
        ('merged_text_components', 'Merged Text Components'),
        ('image_components', 'Image Components'),
        ('filtered_image_components', 'Filtered Image Components'),
        ('character_components', 'Character Components'),
        ('filteredCharacters', 'Final Characters')
    ]
    
    for attr, name in comp_attrs:
        comp = getattr(output, attr, None)
        if comp is not None:
            outputs[name] = ("segmentation", comp.segm_img)
    
    return outputs

@st.cache_data
def create_clickable_component_view(_img_pil, _labels, comp_id, max_display_size=1200):
    """Create interactive component viz with downsampling"""
    img_array = np.array(_img_pil)
    h, w = img_array.shape[:2]
    scale = min(1.0, max_display_size / max(h, w))
    
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = np.array(Image.fromarray(img_array).resize((new_w, new_h), Image.Resampling.LANCZOS))
    else:
        display_img = img_array
    
    # Create overlay
    overlay = display_img.astype(float) * 0.5
    mask = (_labels == comp_id)
    
    if scale < 1.0:
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.Resampling.NEAREST)) > 128
        overlay[mask_resized] = display_img[mask_resized] * 0.7 + np.array([255, 0, 0]) * 0.3
    else:
        overlay[mask] = display_img[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
    
    overlay = overlay.astype(np.uint8)
    
    fig = go.Figure(data=go.Image(z=overlay))
    fig.update_layout(
        height=min(600, int(overlay.shape[0] * 1.1)),
        xaxis=dict(visible=False, range=[0, overlay.shape[1]]),
        yaxis=dict(visible=False, range=[overlay.shape[0], 0], scaleanchor="x"),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode=False
    )
    
    return fig, scale

def create_heatmap(score_map, title):
    """Create interactive plotly heatmap"""
    data = score_map.cpu().numpy() if hasattr(score_map, 'cpu') else score_map
    data = data.squeeze()
    
    fig = go.Figure(data=go.Heatmap(z=data, colorscale='Hot', colorbar=dict(title="Score")))
    fig.update_layout(title=title, xaxis_title="Width", yaxis_title="Height", height=500)
    return fig

def log_intermediate_result(msg, result=None, result_type=None):
    """Log intermediate results"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.intermediate_logs.append({
        'timestamp': timestamp, 'message': msg, 'result': result, 'result_type': result_type
    })
    
    if not enable_logging:
        return
    
    st.markdown(f"**{timestamp}** - {msg}")
    
    if show_preview and result is not None and result_type == "components" and hasattr(result, 'segm_img'):
        st.image(result.segm_img, caption=f"{msg} ({result.nLabels - 1})", width=400)

# ==================== UI ====================
st.title("📝 Text Detection Pipeline")
st.markdown("Visualize and analyze each stage of the text detection pipeline")

if st.session_state.pipeline_run_count > 0:
    st.info(f"✓ Runs: {st.session_state.pipeline_run_count} | Last: {st.session_state.processing_time:.2f}s")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Pipeline Configuration")
    
    preset = st.selectbox("Preset", list(PRESETS.keys()))
    p = PRESETS[preset]
    
    st.divider()
    
    with st.expander("CRAFT Detection", expanded=True):
        mag_ratio = st.slider("Magnification Ratio", 1.0, 10.0, p["mag"], 0.5)
        canvas_size = st.number_input("Canvas Size", 640, 2560, p["canvas"], 32)
        text_threshold = st.slider("Text Threshold", 0.0, 1.0, p["thresh"], 0.05)
    
    with st.expander("Component Analysis"):
        connectivity = st.selectbox("Connectivity", [4, 8], index=0 if p["connectivity"]==4 else 1)
        min_area = st.number_input("Min Area", 0, 100, p["area"])
        min_aspect = st.slider("Min Aspect Ratio", 0.1, 5.0, p["min_asp"], 0.1)
        max_aspect = st.slider("Max Aspect Ratio", 0.5, 10.0, p["max_asp"], 0.1)
        min_dist = st.slider("Min Distance for Merge", 0.0, 50.0, p["dist"], 1.0)
    
    with st.expander("Image Processing"):
        threshold_method = st.selectbox("Threshold Method", 
            ['otsu', 'li', 'isodata', 'mean', 'triangle', 'minimum', 'yen'],
            index=['otsu', 'li', 'isodata', 'mean', 'triangle', 'minimum', 'yen'].index(p["threshold_method"]))
        
        st.markdown("**Component Filtering**")
        min_img_aspect = st.slider("Min Image Aspect", 1.0, 20.0, p["img_aspect"], 1.0)
        min_axis_length = st.slider("Min Axis Length", 10.0, 100.0, p["axis_length"], 5.0)
        similarity_threshold = st.slider("Similarity Threshold", -50.0, 50.0, p["similarity"], 1.0)
        
        st.markdown("**Character Filtering**")
        col1, col2 = st.columns(2)
        with col1:
            min_box_w = st.number_input("Min Width", 1, 500, p["min_box"][0], 5)
            min_box_h = st.number_input("Min Height", 1, 500, p["min_box"][1], 5)
        with col2:
            max_box_w = st.number_input("Max Width", 1, 500, p["max_box"][0], 10)
            max_box_h = st.number_input("Max Height", 1, 500, p["max_box"][1], 10)
        
        max_char_aspect = st.slider("Max Aspect", 0.5, 5.0, p["char_aspect"], 0.1)
        max_filled_area = st.slider("Max Filled", 0.0, 1.0, p["filled_area"], 0.05)
    
    device = st.selectbox("Device", ['cpu', 'cuda'])
    
    st.divider()
    st.subheader("Logging")
    enable_logging = st.checkbox("Enable Intermediate Logging", value=False)
    show_preview = st.checkbox("Show Live Preview", value=False)
    
    st.divider()
    if st.session_state.intermediate_logs:
        st.metric("Log Entries", len(st.session_state.intermediate_logs))
        if st.button("Clear Logs"):
            st.session_state.intermediate_logs = []
            st.rerun()
    
    if st.button("Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==================== MAIN ====================
uploaded = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded:
    try:
        img = Image.open(uploaded).convert('RGB')
        
        pipeline = load_pipeline(
            craftParams(mag_ratio=mag_ratio, canvas_size=canvas_size),
            craftComponentsParams(
                text_threshold=text_threshold, connectivity=connectivity,
                min_area=min_area, min_aspect_ratio=min_aspect,
                max_aspect_ratio=max_aspect, min_dist=min_dist
            ),
            imageComponentsParams(
                threshold=threshold_method, 
                min_image_component_aspect_ratio=min_img_aspect,
                min_image_component_axis_major_length_criterion=min_axis_length,
                similarity_threshold=similarity_threshold,
                min_box_size=(min_box_w, min_box_h),
                max_box_size=(max_box_w, max_box_h),
                max_aspect_ratio=max_char_aspect,
                max_filled_area_portion=max_filled_area
            ),
            device
        )
        
        if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
            st.session_state.intermediate_logs = []
            
            if enable_logging:
                st.divider()
                st.subheader("Execution Log")
                log_container = st.container()
                
                def callback(msg, data=None, dtype=None):
                    with log_container:
                        log_intermediate_result(msg, data, dtype)
                
                pipeline.set_progress_callback(callback)
            
            with st.spinner("Processing..."):
                try:
                    start = time.time()
                    output = pipeline.forward(img, verbose=True)
                    proc_time = time.time() - start
                    
                    st.session_state.pipeline_output = output
                    st.session_state.processing_time = proc_time
                    st.session_state.pipeline_run_count += 1
                    
                    st.success(f"✓ Completed in {proc_time:.2f}s")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
        
        # Display results
        if st.session_state.pipeline_output:
            output = st.session_state.pipeline_output
            
            st.divider()
            st.subheader("Pipeline Metrics")
            
            # Metrics
            cols = st.columns(5)
            stages = [
                ('score_text_components', 'Initial'),
                ('filtered_text_components', 'Filtered'),
                ('merged_text_components', 'Merged'),
                ('character_components', 'Characters'),
                ('filteredCharacters', 'Final')
            ]
            
            prev = 0
            for i, (attr, label) in enumerate(stages):
                comp = getattr(output, attr, None)
                count = comp.nLabels - 1 if comp is not None else 0
                with cols[i]:
                    delta = count - prev if i > 0 and prev > 0 else None
                    st.metric(label, count, delta=delta)
                prev = count if count > 0 else prev
            
            # Chart
            valid_stages = [(l, getattr(output, a, None)) for a, l in stages]
            valid_stages = [(l, c.nLabels-1) for l, c in valid_stages if c is not None]
            
            if valid_stages:
                labels, counts = zip(*valid_stages)
                fig = go.Figure(
                    data=go.Bar(x=labels, y=counts, text=counts, textposition='auto',
                               marker=dict(color=counts, colorscale='Viridis', showscale=True)),
                    layout=dict(title="Component Progression", height=350, showlegend=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Inspector", "Comparison", "Export"])
            
            # TAB 1: OVERVIEW
            with tab1:
                st.subheader("Pipeline Stages")
                stage_tabs = st.tabs(["Original", "Binary", "CRAFT", "Text", "Image", "Final"])
                
                with stage_tabs[0]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(output.img_pil, use_container_width=True)
                    with col2:
                        st.metric("Width", f"{output.img_pil.size[0]}px")
                        st.metric("Height", f"{output.img_pil.size[1]}px")
                
                with stage_tabs[1]:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(output.binary_img, cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                
                with stage_tabs[2]:
                    st.plotly_chart(create_heatmap(output.score_text, "CRAFT Score"), use_container_width=True)
                
                with stage_tabs[3]:
                    for comp, name, desc in [
                        (output.score_text_components, "Initial", "Raw CRAFT"),
                        (output.filtered_text_components, "Filtered", "After filtering"),
                        (output.merged_text_components, "Merged", "After merging")
                    ]:
                        if comp is not None:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.image(comp.segm_img, use_container_width=True)
                            with col2:
                                st.metric("Components", comp.nLabels - 1)
                                st.caption(desc)
                            st.divider()
                
                with stage_tabs[4]:
                    for comp, name, desc in [
                        (output.image_components, "Image", "From binary"),
                        (output.filtered_image_components, "Filtered", "Lines removed"),
                        (output.character_components, "Characters", "Associated with text")
                    ]:
                        if comp is not None:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.image(comp.segm_img, use_container_width=True)
                            with col2:
                                st.metric("Components", comp.nLabels - 1)
                                st.caption(desc)
                            st.divider()
                
                with stage_tabs[5]:
                    cols = st.columns(2)
                    with cols[0]:
                        if output.character_components is not None:
                            st.image(output.character_components.segm_img, use_container_width=True)
                            st.caption(f"Characters: {output.character_components.nLabels-1}")
                    with cols[1]:
                        if output.filteredCharacters is not None:
                            st.image(output.filteredCharacters.segm_img, use_container_width=True)
                            st.caption(f"Final: {output.filteredCharacters.nLabels-1}")
            
            # TAB 2: INSPECTOR
            with tab2:
                st.subheader("Component Inspector")
                
                target = st.radio("Analyze:", ["Characters", "Final"], horizontal=True, key="inspect")
                comp_set = output.filteredCharacters if target == "Final" else output.character_components
                
                if comp_set is not None and comp_set.nLabels > 1:
                    if 'selected_component' not in st.session_state:
                        st.session_state.selected_component = 1
                    
                    st.info("💡 Click on a component to select it")
                    
                    fig, scale = create_clickable_component_view(
                        output.img_pil, comp_set.labels, st.session_state.selected_component
                    )
                    
                    selected = st.plotly_chart(fig, use_container_width=True, 
                                              on_select="rerun", selection_mode="points")
                    
                    if selected and selected.selection and selected.selection.points:
                        point = selected.selection.points[0]
                        x, y = int(point['x'] / scale), int(point['y'] / scale)
                        
                        if 0 <= y < comp_set.labels.shape[0] and 0 <= x < comp_set.labels.shape[1]:
                            clicked_label = comp_set.labels[y, x]
                            if clicked_label > 0:
                                st.session_state.selected_component = int(clicked_label)
                    
                    comp_id = st.slider("Component ID", 1, comp_set.nLabels - 1, 
                                       st.session_state.selected_component, key="slider")
                    
                    if comp_id != st.session_state.selected_component:
                        st.session_state.selected_component = comp_id
                    
                    st.divider()
                    stats = get_component_stats(comp_set.labels, st.session_state.selected_component)
                    
                    if stats is not None:
                        cols = st.columns(5)
                        metrics = [("ID", f"#{stats['ID']}"), ("Pixels", f"{stats['Pixels']:,}"),
                                  ("Width", f"{stats['Width']}px"), ("Height", f"{stats['Height']}px"),
                                  ("Aspect", f"{stats['Width']/stats['Height']:.2f}" if stats['Height']>0 else "N/A")]
                        
                        for col, (label, value) in zip(cols, metrics):
                            with col:
                                st.metric(label, value)
                        
                        st.caption(f"Position: ({stats['X']}, {stats['Y']}) → "
                                 f"({stats['X']+stats['Width']}, {stats['Y']+stats['Height']})")
                    
                    st.divider()
                    st.subheader("All Components Summary")
                    
                    all_stats = get_all_component_stats(comp_set.labels, comp_set.nLabels)
                    
                    if all_stats:
                        df = pd.DataFrame(all_stats)
                        df['Aspect'] = df['Width'] / df['Height']
                        
                        cols = st.columns(4)
                        for col, (label, val) in zip(cols, [
                            ("Total", len(all_stats)),
                            ("Avg Width", f"{df['Width'].mean():.1f}px"),
                            ("Avg Height", f"{df['Height'].mean():.1f}px"),
                            ("Avg Aspect", f"{df['Aspect'].mean():.2f}")
                        ]):
                            with col:
                                st.metric(label, val)
                        
                        st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.info("No components to inspect")
            
            # TAB 3: COMPARISON
            with tab3:
                st.subheader("Stage Comparison")
                
                view = st.radio("View:", ["Any vs Any", "Quick Compare", "Overlay"], horizontal=True)
                
                if view == "Any vs Any":
                    all_outputs = get_all_pipeline_outputs(output)
                    names = list(all_outputs.keys())
                    
                    col1, col2 = st.columns(2)
                    
                    for col, key, idx in [(col1, "left", 0), (col2, "right", min(len(names)-1, 1))]:
                        with col:
                            name = st.selectbox(f"{key.title()}:", names, index=idx, key=f"{key}_sel")
                            img_type, img = all_outputs[name]
                            
                            if img_type == "image":
                                st.image(img, use_container_width=True)
                            elif img_type == "gray":
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.imshow(img, cmap='gray')
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)
                            elif img_type == "heatmap":
                                st.plotly_chart(create_heatmap(img, name), use_container_width=True)
                            else:
                                st.image(img, use_container_width=True)
                            
                            shape = np.array(img).shape if not isinstance(img, Image.Image) else img.size
                            st.caption(f"{name} - Shape: {shape}")
                
                elif view == "Quick Compare":
                    mode = st.selectbox("Mode:", [
                        "Original vs Final", "Binary vs Characters",
                        "Text: Initial/Filtered/Merged", "Image: All/Filtered/Characters"
                    ])
                    
                    if mode == "Original vs Final":
                        cols = st.columns(2)
                        with cols[0]:
                            st.image(output.img_pil, use_container_width=True, caption="Original")
                        with cols[1]:
                            if output.filteredCharacters:
                                st.image(output.filteredCharacters.segm_img, use_container_width=True,
                                       caption=f"Final ({output.filteredCharacters.nLabels-1})")
                    
                    elif mode == "Binary vs Characters":
                        cols = st.columns(2)
                        with cols[0]:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.imshow(output.binary_img, cmap='gray')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                        with cols[1]:
                            if output.character_components:
                                st.image(output.character_components.segm_img, use_container_width=True)
                    
                    else:
                        stages_map = {
                            "Text: Initial/Filtered/Merged": [
                                (output.score_text_components, "Initial"),
                                (output.filtered_text_components, "Filtered"),
                                (output.merged_text_components, "Merged")
                            ],
                            "Image: All/Filtered/Characters": [
                                (output.image_components, "All"),
                                (output.filtered_image_components, "Filtered"),
                                (output.character_components, "Characters")
                            ]
                        }
                        
                        stages = stages_map[mode]
                        cols = st.columns(3)
                        for col, (comp, label) in zip(cols, stages):
                            with col:
                                if comp:
                                    st.image(comp.segm_img, use_container_width=True)
                                    st.caption(f"{label}: {comp.nLabels-1}")
                
                else:  # Overlay
                    all_outputs = get_all_pipeline_outputs(output)
                    overlay_opts = {k: v for k, v in all_outputs.items() if k != "Original Image"}
                    
                    name = st.selectbox("Overlay:", list(overlay_opts.keys()))
                    img_type, overlay_img = overlay_opts[name]
                    opacity = st.slider("Opacity", 0.0, 1.0, 0.4, 0.05)
                    
                    base = np.array(output.img_pil)
                    
                    fig, ax = plt.subplots(figsize=(14, 10))
                    ax.imshow(base, alpha=0.6)
                    
                    if img_type == "gray":
                        resized = resize_to_match(overlay_img, base.shape)
                        ax.imshow(resized, cmap='RdYlBu_r', alpha=opacity)
                    elif img_type == "heatmap":
                        resized = resize_to_match(overlay_img, base.shape)
                        ax.imshow(resized, cmap='hot', alpha=opacity)
                    else:
                        resized = resize_to_match(overlay_img, base.shape, is_label_map=True)
                        ax.imshow(resized, alpha=opacity)
                    
                    ax.axis('off')
                    ax.set_title(f"{name} overlay", fontsize=16, pad=20)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.caption(f"Base: {base.shape}, Overlay: {np.array(overlay_img).shape} → Resized")
            
            # TAB 4: EXPORT
            with tab4:
                st.subheader("Export Results")
                
                cols = st.columns(2)
                
                with cols[0]:
                    st.markdown("**Images**")
                    exports = [
                        ("Original", output.img_pil),
                        ("Binary", output.binary_img),
                        ("Score_Text", output.score_text_components.segm_img if output.score_text_components else None),
                        ("Filtered_Text", output.filtered_text_components.segm_img if output.filtered_text_components else None),
                        ("Merged_Text", output.merged_text_components.segm_img if output.merged_text_components else None),
                        ("Image_Components", output.image_components.segm_img if output.image_components else None),
                        ("Filtered_Image", output.filtered_image_components.segm_img if output.filtered_image_components else None),
                        ("Characters", output.character_components.segm_img if output.character_components else None),
                        ("Final", output.filteredCharacters.segm_img if output.filteredCharacters else None)
                    ]
                    
                    for name, img in exports:
                        if img is not None:
                            buf = io.BytesIO()
                            if isinstance(img, Image.Image):
                                img.save(buf, format='PNG')
                            else:
                                fig, ax = plt.subplots()
                                ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                                ax.axis('off')
                                fig.savefig(buf, format='png', bbox_inches='tight')
                                plt.close(fig)
                            
                            st.download_button(
                                f"📥 {name.replace('_', ' ')}",
                                buf.getvalue(),
                                f"{name.lower()}.png",
                                key=f"dl_{name}"
                            )
                
                with cols[1]:
                    st.markdown("**Configuration**")
                    config = {
                        'preset': preset,
                        'mag_ratio': mag_ratio,
                        'canvas_size': canvas_size,
                        'text_threshold': text_threshold,
                        'connectivity': connectivity,
                        'min_area': min_area,
                        'min_aspect_ratio': min_aspect,
                        'max_aspect_ratio': max_aspect,
                        'min_dist': min_dist,
                        'threshold_method': threshold_method,
                        'min_image_aspect': min_img_aspect,
                        'min_axis_length': min_axis_length,
                        'similarity_threshold': similarity_threshold,
                        'min_box_size': [min_box_w, min_box_h],
                        'max_box_size': [max_box_w, max_box_h],
                        'max_char_aspect': max_char_aspect,
                        'max_filled_area': max_filled_area,
                        'processing_time': st.session_state.processing_time
                    }
                    
                    st.download_button(
                        "📄 Download Config",
                        pd.DataFrame([config]).to_json(indent=2),
                        "config.json",
                        mime="application/json"
                    )
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("👆 Upload an image to begin")