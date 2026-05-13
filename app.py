import streamlit as st
import requests
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Satellite Change Detector",
    page_icon="🛰️",
    layout="wide"
)

st.title("Satellite Temporal Change Detector")

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Change",
    "🗺️ Global Change Map",
    "ℹ️ Model Info"
])

CHANGE_COLORS = {
    'deforestation': '#e74c3c',
    'flood':         '#3498db',
    'construction':  '#e67e22',
    'agricultural':  '#27ae60',
    'no_change':     '#95a5a6'
}

CHANGE_ICONS = {
    'deforestation': '🌲',
    'flood':         '🌊',
    'construction':  '🏗️',
    'agricultural':  '🌾',
    'no_change':     '✅'
}


def render_change_result(result):
    ct     = result['change_type']
    color  = CHANGE_COLORS.get(ct, '#333')
    icon   = CHANGE_ICONS.get(ct, '❓')
    sev    = result['severity']

    st.markdown(
        f"<div style='background:{color};padding:16px;"
        f"border-radius:10px;text-align:center;"
        f"color:white;font-size:22px;font-weight:700'>"
        f"{icon} {ct.replace('_',' ').upper()} DETECTED"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Change Type",      ct.replace('_', ' ').title())
    c2.metric("Confidence",       f"{result['confidence']*100:.1f}%")
    c3.metric("Severity",         sev)
    c4.metric("Pixels Changed",   f"{result['pixel_change_pct']:.1f}%")

    st.info(f"**Recommendation:** {result['recommendation']}")
    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Class Probabilities")
        probs  = result['class_probabilities']
        labels = list(probs.keys())
        values = list(probs.values())
        colors = [CHANGE_COLORS.get(l, '#333') for l in labels]

        fig = go.Figure(go.Bar(
            x      = values,
            y      = labels,
            orientation = 'h',
            marker_color= colors,
            text   = [f"{v*100:.1f}%" for v in values],
            textposition = 'outside'
        ))
        fig.update_layout(
            height=80,
            margin=dict(t=20, b=20, l=20, r=60),
            xaxis=dict(range=[0, 1.1])
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Change Indices")
        ndvi  = result['ndvi_change']
        water = result['water_change']

        st.metric(
            "NDVI Change (Vegetation)",
            f"{ndvi:+.4f}",
            delta=f"{'Loss' if ndvi < 0 else 'Gain'}",
            delta_color="inverse" if ndvi < 0 else "normal"
        )
        st.metric(
            "Water Index Change",
            f"{water:+.4f}",
            delta=f"{'Increase' if water > 0 else 'Decrease'}",
            delta_color="inverse" if water > 0 else "normal"
        )
        st.metric(
            "Area Description",
            result['area_description']
        )


#  Tab 1: Analyse Change 
with tab1:
    st.subheader("Upload Before & After Satellite Images")
    col_a, col_b = st.columns(2)

    with col_a:
        before_file = st.file_uploader(
            "📅 BEFORE image", type=["png","jpg","jpeg"],
            key="before"
        )
        if before_file:
            st.image(before_file, caption="Before", use_column_width=True)

    with col_b:
        after_file = st.file_uploader(
            "📅 AFTER image", type=["png","jpg","jpeg"],
            key="after"
        )
        if after_file:
            st.image(after_file, caption="After", use_column_width=True)

    if before_file and after_file:
        if st.button("🔍 Detect Changes",
                     type="primary", use_container_width=True):
            with st.spinner("Analysing satellite imagery..."):
                try:
                    files  = {
                        "before": (before_file.name,
                                   before_file.getvalue(),
                                   before_file.type),
                        "after":  (after_file.name,
                                   after_file.getvalue(),
                                   after_file.type)
                    }
                    resp   = requests.post(
                        f"{API_URL}/detect", files=files
                    )
                    result = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            st.divider()
            render_change_result(result)

            # Difference heatmap
            st.subheader("Change Heatmap")
            before_arr = np.array(
                Image.open(before_file).convert('RGB').resize((224,224))
            )
            after_arr  = np.array(
                Image.open(after_file).convert('RGB').resize((224,224))
            )
            diff = np.abs(
                before_arr.astype(np.float32) -
                after_arr.astype(np.float32)
            ).mean(axis=2)
            diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

            fig = px.imshow(
                diff_norm,
                color_continuous_scale='hot',
                title="Pixel-level Change Heatmap (bright = high change)"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload both a **Before** and **After** satellite image to detect changes.")


#  Tab 2: Global Map 
with tab2:
    st.subheader("Global Change Detection Map")
    try:
        detections = requests.get(f"{API_URL}/map/detections").json()
        df = pd.DataFrame(detections)

        # Summary metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Detections", len(df))
        c2.metric("🌲 Deforestation", len(df[df['change_type']=='deforestation']))
        c3.metric("🌊 Floods",        len(df[df['change_type']=='flood']))
        c4.metric("🏗️ Construction",  len(df[df['change_type']=='construction']))
        c5.metric("🌾 Agricultural",  len(df[df['change_type']=='agricultural']))

        # Map
        fig = px.scatter_geo(
            df,
            lat='lat', lon='lon',
            color='change_type',
            size='area_ha',
            size_max=30,
            hover_name='location',
            hover_data={
                'confidence': ':.1%',
                'area_ha':    ':,.1f',
                'date_before':True,
                'date_after': True
            },
            color_discrete_map=CHANGE_COLORS,
            title="Global Satellite Change Detections",
            projection='natural earth'
        )
        fig.update_layout(
            height=500,
            margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Detection Details")
        st.dataframe(
            df[['location','change_type','confidence',
                'area_ha','date_before','date_after']]
            .sort_values('confidence', ascending=False),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Could not load map data: {e}")


# Tab 3: Model Info 
with tab3:
    st.subheader("Model Architecture")
    try:
        info = requests.get(f"{API_URL}/model/info").json()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("CNN Model",    info['cnn_model'])
        c2.metric("XGB Model",    info['xgb_model'])
        c3.metric("CNN Accuracy", f"{info['cnn_accuracy']*100:.1f}%")
        c4.metric("XGB Accuracy", f"{info['xgb_accuracy']*100:.1f}%")

        st.divider()
        st.markdown("""
                    """)
    except Exception:
        st.error("Cannot connect to API.")