# ===================== IMPORTS =====================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import time
from io import BytesIO
from faker import Faker

# ===================== CONFIGURATION =====================
st.set_page_config(
    page_title="Incredible India Tourism Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM STYLING =====================
def inject_css():
    st.markdown("""
    <style>
    :root {
        --primary: #FF9933;
        --secondary: #138808;
        --bg: #F0F2F6;
        --text: #262730;
    }
    
    .hero {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 5rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    .testimonial {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .pulse {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: var(--primary);
        margin: 0 5px;
        animation: pulse 1.5s infinite;
        animation-delay: calc(var(--i) * 0.1s);
    }
    
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.7; }
        70% { transform: scale(1.3); opacity: 0.3; }
        100% { transform: scale(0.8); opacity: 0.7; }
    }
    
    /* Dark mode toggle */
    [data-testid="stSidebar"] {
        background: #f0f2f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== DATA LOADERS =====================
@st.cache_data
def load_data():
    """Load the India tourism dataset"""
    df = pd.read_csv(r"C:\Users\cmsan\OneDrive\Desktop\india tourism\india tourism.csv")
    
    # Clean and preprocess data
    df = df[df['Circle'] != 'Total']  # Remove summary rows
    df = df[df['Name of the Monument '] != 'Grand Total']
    
    # Calculate additional metrics
    df['Total_2019'] = df['Domestic-2019-20'] + df['Foreign-2019-20']
    df['Total_2020'] = df['Domestic-2020-21'] + df['Foreign-2020-21']
    df['Growth_Rate'] = ((df['Total_2020'] - df['Total_2019']) / df['Total_2019']) * 100
    
    # Add sample coordinates for mapping (in a real app, you'd have precise coordinates)
    fake = Faker()
    np.random.seed(42)
    
    # Assign approximate coordinates based on Circle (region)
    circle_coords = {
        'Agra': (27.1767, 78.0081),
        'Lucknow': (26.8467, 80.9462),
        'Jhansi': (25.4484, 78.5685),
        'Sarnath': (25.3755, 83.0228),
        'Thrissur': (10.5276, 76.2144),
        'Chennai': (13.0827, 80.2707),
        'Tiruchirappalli': (10.7905, 78.7047),
        'Bhopal': (23.2599, 77.4126),
        'Jabalpur': (23.1815, 79.9864),
        'Dharwad': (15.4589, 75.0078),
        'Hampi': (15.3350, 76.4600),
        'Banglore': (12.9716, 77.5946),
        'Raiganj': (25.6133, 88.1198),
        'Kolkata': (22.5726, 88.3639),
        'Rajkot': (22.3039, 70.8022),
        'Vadodara': (22.3072, 73.1812),
        'Bhubaneswar': (20.2961, 85.8245),
        'Aurangabad': (19.8762, 75.3433),
        'Mumbai': (19.0760, 72.8777),
        'Nagpur': (21.1458, 79.0882),
        'Chandigarh': (30.7333, 76.7794),
        'Delhi': (28.6139, 77.2090),
        'Guwahati': (26.1445, 91.7362),
        'Goa': (15.2993, 74.1240),
        'Hyderabad': (17.3850, 78.4867),
        'Jaipur': (26.9124, 75.7873),
        'Jodhpur': (26.2389, 73.0243),
        'Leh': (34.1526, 77.5771),
        'Patna': (25.5941, 85.1376),
        'Raipur': (21.2514, 81.6296),
        'Shimla': (31.1048, 77.1734),
        'Srinagar': (34.0836, 74.7973),
        'Amaravati': (16.5726, 80.3573)
    }
    
    df['Latitude'] = df['Circle'].map(lambda x: circle_coords.get(x, (0, 0))[0])
    df['Longitude'] = df['Circle'].map(lambda x: circle_coords.get(x, (0, 0))[1])
    
    # Add small random offsets to prevent overlapping markers
    df['Latitude'] = df['Latitude'] + np.random.uniform(-0.2, 0.2, size=len(df))
    df['Longitude'] = df['Longitude'] + np.random.uniform(-0.2, 0.2, size=len(df))
    
    return df

@st.cache_data
def load_sample_reviews():
    """Generate sample tourist reviews"""
    reviews = [
        "The Taj Mahal was breathtaking! Worth every minute of the long journey.",
        "Hampi's ruins transport you back in time. Magical experience!",
        "Khajuraho temples have incredible carvings but need better maintenance.",
        "The Sun Temple architecture is stunning but the location is remote.",
        "Ajanta caves are a masterpiece of ancient Indian art.",
        "Ellora caves showcase India's rich cultural heritage beautifully.",
        "The beaches of Goa are paradise but can get overcrowded.",
        "Kerala's backwaters offer a serene and unique experience.",
        "The forts of Rajasthan are magnificent but need better maintenance.",
        "Varanasi's ghats provide a spiritual experience like no other."
    ]
    return reviews

# ===================== COMPONENTS =====================
def hero_section():
    """Animated hero section with loading effect"""
    with st.container():
        st.markdown("""
        <div class="hero">
            <h1 class="hero-title">Incredible India</h1>
            <p class="hero-subtitle">Explore India's Rich Cultural Heritage</p>
            <div style="margin-top: 1rem;">
                <span class="pulse" style="--i:0"></span>
                <span class="pulse" style="--i:1"></span>
                <span class="pulse" style="--i:2"></span>
                <span class="pulse" style="--i:3"></span>
                <span class="pulse" style="--i:4"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate loading
        with st.spinner("Loading dashboard components..."):
            time.sleep(1)

def metric_card(title, value, icon, color):
    """Beautiful metric card with icons"""
    st.markdown(f"""
    <div class="metric-card" style="border-left: 5px solid {color}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-content">
            <h3>{title}</h3>
            <h2>{value}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===================== PAGES =====================
def home_page():
    """Home page with overview content"""
    hero_section()
    df = load_data()
    
    # Key metrics
    st.subheader("🏛️ India Tourism Overview (2019-2021)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_card("Total Monuments", len(df), "🏛️", "#FF9933")
    with col2:
        metric_card("2019 Visitors", f"{df['Total_2019'].sum()/1000000:.1f}M", "👥", "#138808")
    with col3:
        top_circle = df.groupby('Circle')['Total_2019'].sum().idxmax()
        metric_card("Top Region", top_circle, "📍", "#000080")
    with col4:
        metric_card("Avg Growth", f"{df['Growth_Rate'].mean():.1f}%", "📈", "#FF0000")
    
    # Introduction
    st.markdown("""
    ## Welcome to the India Tourism Analytics Dashboard
    
    This interactive dashboard provides comprehensive insights into India's tourism landscape, featuring:
    - **Monument Analytics**: Visitor patterns across India's heritage sites
    - **Geospatial Analysis**: Explore tourism hotspots across the country
    - **Image Processing**: Enhance your tourism photos
    - **Sentiment Analysis**: Understand visitor experiences
    """)
    
    # Quick charts
    st.subheader("Quick Insights")
    tab1, tab2, tab3 = st.tabs(["Visitor Distribution", "Foreign vs Domestic", "Growth Analysis"])
    
    with tab1:
        circle_totals = df.groupby('Circle')['Total_2019'].sum().reset_index()
        fig = px.pie(circle_totals, values='Total_2019', names='Circle', 
                    title='Visitor Distribution by Region (2019)',
                    color_discrete_sequence=px.colors.sequential.Sunsetdark)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.bar(df.nlargest(10, 'Total_2019'), 
                    x='Name of the Monument ', 
                    y=['Domestic-2019-20', 'Foreign-2019-20'],
                    title='Top 10 Monuments: Domestic vs Foreign Visitors (2019)',
                    labels={'value': 'Visitors', 'variable': 'Visitor Type'},
                    barmode='group',
                    color_discrete_sequence=['#FF9933', '#138808'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.scatter(df.nlargest(20, 'Total_2019'), 
                        x='Total_2019', 
                        y='Growth_Rate', 
                        color='Circle',
                        size='Foreign-2019-20', 
                        hover_name='Name of the Monument ',
                        title='Growth Rate vs Total Visitors (Top 20 Monuments)',
                        labels={'Total_2019': 'Total Visitors (2019)', 'Growth_Rate': 'Growth Rate (%)'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

def show_3d_visualizations():
    """Interactive 3D analytics"""
    st.header("📊 3D Tourism Insights")
    df = load_data()
    
    with st.expander("About these visualizations"):
        st.write("""
        These 3D visualizations help understand tourism patterns across India's heritage sites.
        - Use the controls below to customize views
        - Hover over data points for details
        - Rotate plots by dragging with your mouse
        """)
    
    # 3D Chart Selector
    chart_type = st.selectbox(
        "Select 3D Visualization Type",
        ["Visitor Distribution", "Growth Patterns", "Regional Comparison"]
    )
    
    if chart_type == "Visitor Distribution":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get top 20 monuments for better visualization
        top_df = df.nlargest(20, 'Total_2019')
        
        x = range(len(top_df))
        y = top_df['Foreign-2019-20'] / 1000
        z = np.zeros(len(top_df))
        dx = np.ones(len(top_df)) * 0.5
        dy = np.ones(len(top_df)) * 0.5
        dz = top_df['Domestic-2019-20'] / 1000
        
        ax.bar3d(x, y, z, dx, dy, dz, shade=True, 
                color=plt.cm.viridis(top_df['Total_2019']/top_df['Total_2019'].max()))
        ax.set_xticks(x)
        ax.set_xticklabels(top_df['Name of the Monument '], rotation=90)
        ax.set_xlabel('Monuments')
        ax.set_ylabel('Foreign Visitors (Thousands)')
        ax.set_zlabel('Domestic Visitors (Thousands)')
        ax.set_title('Top Monuments by Visitor Volume (3D Bars)')
        st.pyplot(fig)
        
    elif chart_type == "Growth Patterns":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        top_df = df.nlargest(30, 'Total_2019')
        
        scatter = ax.scatter(
            top_df['Domestic-2019-20'] / 1000,
            top_df['Foreign-2019-20'] / 1000,
            top_df['Total_2019'] / 1000,
            c=top_df['Growth_Rate'],
            cmap='viridis',
            s=top_df['Total_2019']/50000,
            alpha=0.7
        )
        
        ax.set_xlabel('Domestic (Thousands)')
        ax.set_ylabel('Foreign (Thousands)')
        ax.set_zlabel('Total Visitors (Thousands)')
        ax.set_title('Visitor Growth Patterns (Size = Popularity, Color = Growth Rate)')
        fig.colorbar(scatter, ax=ax, label='Growth Rate (%)')
        
        # Add monument names as annotations
        for i, row in top_df.iterrows():
            ax.text(row['Domestic-2019-20']/1000, 
                   row['Foreign-2019-20']/1000, 
                   row['Total_2019']/1000, 
                   row['Name of the Monument '][:15] + '...', 
                   size=8)
        
        st.pyplot(fig)
        
    elif chart_type == "Regional Comparison":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        selected_circles = st.multiselect(
            "Select Regions to Compare",
            df['Circle'].unique(),
            default=['Agra', 'Delhi', 'Mumbai']
        )
        
        for circle in selected_circles:
            circle_data = df[df['Circle'] == circle]
            ax.plot(
                circle_data['Domestic-2019-20'] / 1000,
                circle_data['Foreign-2019-20'] / 1000,
                circle_data['Total_2019'] / 1000,
                'o-',
                label=circle
            )
        
        ax.set_xlabel('Domestic (Thousands)')
        ax.set_ylabel('Foreign (Thousands)')
        ax.set_zlabel('Total (Thousands)')
        ax.set_title('Region-wise Visitor Comparison')
        ax.legend()
        st.pyplot(fig)

def show_geospatial():
    """Interactive map visualizations"""
    st.header("🗺️ Geospatial Explorer")
    df = load_data()
    
    with st.expander("About these maps"):
        st.write("""
        Explore India's tourism geography through interactive maps:
        - **Choropleth**: Region-wise visitor density
        - **Heatmap**: Monument visitor concentration
        - **Bubble**: Individual monument details
        """)
    
    # Map Type Selector
    map_type = st.radio(
        "Map Visualization",
        ["Choropleth", "Heatmap", "Bubble"],
        horizontal=True
    )
    
    # Central coordinates for India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='cartodbpositron')
    
    if map_type == "Choropleth":
        # Region-level data
        region_data = df.groupby('Circle').agg({
            'Total_2019': 'sum',
            'Growth_Rate': 'mean'
        }).reset_index()
        
        # Load India states GeoJSON (simplified)
        india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
        
        folium.Choropleth(
            geo_data=india_geojson,
            name='choropleth',
            data=region_data,
            columns=['Circle', 'Total_2019'],
            key_on='feature.properties.NAME_1',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Total Visitors (2019)',
            highlight=True
        ).add_to(m)
        
        # Add tooltips
        folium.GeoJson(
            india_geojson,
            name='Labels',
            style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
            tooltip=folium.features.GeoJsonTooltip(
                fields=['NAME_1'],
                aliases=['State: '],
                labels=True,
                sticky=False
            )
        ).add_to(m)
        
    elif map_type == "Heatmap":
        heat_data = df[['Latitude', 'Longitude', 'Total_2019']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)
        
    elif map_type == "Bubble":
        mc = MarkerCluster()
        
        for _, row in df.iterrows():
            popup_content = f"""
            <b>{row['Name of the Monument ']}</b><br>
            Region: {row['Circle']}<br>
            2019 Visitors: {row['Total_2019']:,}<br>
            Domestic: {row['Domestic-2019-20']:,}<br>
            Foreign: {row['Foreign-2019-20']:,}<br>
            Growth: {row['Growth_Rate']:.1f}%
            """
            
            # Different icons for different types of monuments
            icon_type = "monument"
            if "Temple" in row['Name of the Monument ']:
                icon_type = "temple"
            elif "Fort" in row['Name of the Monument ']:
                icon_type = "fort"
            elif "Cave" in row['Name of the Monument ']:
                icon_type = "cave"
            
            # Create custom icon
            icon = folium.Icon(
                color='blue',
                icon_color='white',
                icon=icon_type,
                prefix='fa'
            ) if icon_type != "monument" else folium.Icon(color='red')
            
            mc.add_child(
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=popup_content,
                    icon=icon
                )
            )
        
        m.add_child(mc)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)

def show_image_processing():
    """PIL Image Processing Section"""
    st.header("🖼️ Image Processing Lab")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.warning("Please upload an image to proceed")
        return
    
    image = Image.open(uploaded_file)
    
    # Processing options
    st.subheader("Enhancement Tools")
    col1, col2 = st.columns(2)
    
    with col1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
    
    with col2:
        color = st.slider("Color", 0.0, 2.0, 1.0, 0.1)
        filter_type = st.selectbox("Filter", 
                                 ["None", "Blur", "Contour", "Detail", "Edge Enhance", "Sepia"])
    
    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color)
    
    # Apply filters
    if filter_type == "Blur":
        image = image.filter(ImageFilter.BLUR)
    elif filter_type == "Contour":
        image = image.filter(ImageFilter.CONTOUR)
    elif filter_type == "Detail":
        image = image.filter(ImageFilter.DETAIL)
    elif filter_type == "Edge Enhance":
        image = image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "Sepia":
        # Convert to sepia
        sepia_filter = ImageOps.colorize(
            image.convert("L"), 
            "#704214", "#C0A080"
        )
        image = Image.blend(image, sepia_filter, 0.5)
    
    # Display results
    st.subheader("Processed Image")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Enhanced Image", use_column_width=True)
    
    with col2:
        # Edge detection for segmentation
        st.write("**Image Segmentation (Edge Detection)**")
        gray_image = image.convert('L')
        edges = gray_image.filter(ImageFilter.FIND_EDGES)
        st.image(edges, caption="Edge Detection", use_column_width=True)
    
    # Download button
    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Enhanced Image",
        data=byte_im,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )

def show_text_analysis():
    """Text Analysis Section"""
    st.header("📝 Visitor Sentiment Analysis")
    reviews = load_sample_reviews()
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    
    st.subheader("Sample Reviews")
    selected_review = st.selectbox("Select a review to analyze:", reviews)
    
    # Analyze sentiment
    sentiment = sia.polarity_scores(selected_review)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Scores")
        st.write(f"**Compound:** {sentiment['compound']:.2f}")
        st.write(f"**Positive:** {sentiment['pos']:.2f}")
        st.write(f"**Negative:** {sentiment['neg']:.2f}")
        st.write(f"**Neutral:** {sentiment['neu']:.2f}")
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment['compound'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Meter"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, 0], 'color': "orange"},
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Word Cloud")
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='autumn').generate(selected_review)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    # All reviews analysis
    st.subheader("All Reviews Analysis")
    all_sentiments = [sia.polarity_scores(review) for review in reviews]
    sentiment_df = pd.DataFrame(all_sentiments)
    sentiment_df['Review'] = reviews
    
    fig = px.bar(sentiment_df, 
                x='Review', 
                y=['pos', 'neg', 'neu'],
                title='Sentiment Distribution Across Reviews',
                labels={'value': 'Score', 'variable': 'Sentiment'},
                color_discrete_sequence=['#138808', '#FF0000', '#FF9933'])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ===================== MAIN APP =====================
def main():
    # Inject custom CSS
    inject_css()
    
    # Navigation
    with st.sidebar:
        st.image(r"C:\Users\cmsan\OneDrive\Desktop\india tourism\india.png", use_column_width=True)
        selected = option_menu(
            menu_title="Explore India",
            options=["🏠 Home", "📊 3D Insights", "🗺️ Geospatial", 
                    "🖼️ Image Lab", "📝 Sentiment"],
            icons=["house", "bar-chart", "map", 
                  "image", "chat"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#FF9933"},
            }
        )
    
    # Page router
    if selected == "🏠 Home":
        home_page()
    elif selected == "📊 3D Insights":
        show_3d_visualizations()
    elif selected == "🗺️ Geospatial":
        show_geospatial()
    elif selected == "🖼️ Image Lab":
        show_image_processing()
    elif selected == "📝 Sentiment":
        show_text_analysis()

# ===================== RUN APP =====================
if __name__ == "__main__":
    import requests  # Import requests for sample images
    main()