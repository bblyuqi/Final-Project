import streamlit as st
import pandas as pd
import numpy as np
import time
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, sqrt, asin
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Yogyakarta",
    page_icon="🏞️",
    layout="wide"
)

st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                    url("https://drive.google.com/thumbnail?id=1lBERqqOhsuYOHQvazToiqpglBIyx49Pj&sz=w2000");
        background-size: contain;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    
    /* Container untuk header */
    .header-container {
        background-color:  #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #FF9800;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
    }
    
    /* Container untuk konten */
    .content-container {
        background-color: rgba(52, 73, 94, 1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.8);
    }
    
    /* Global text color */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp div, .stApp span, .stApp label, .stApp a {
        color: white !important;
    }
    
    /* Specific container text color */
    .content-container span, .content-container label, 
    .content-container div, .content-container h1, .content-container h2, 
    .content-container h3, .header-container p, .header-container h1, 
    .header-container h2, .header-container h3 {
        color: white !important;
    }
    
    /* Styling untuk tab */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(52, 73, 94, 1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #3498db;
        border-radius: 5px;
        padding: 8px 16px;
        color: white !important;
    }
    
    /* Styling untuk sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(52, 73, 94, 1);
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] div {
        color: white !important;
    }
    
    /* Fix specifically for the sidebar header */
    [data-testid="stSidebar"] header {
        color: white !important;
    }
    
    /* Bentuk dan warna tombol */
    .stButton > button {
        background-color: #FF9800;
        color: white !important;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #F57C00;
    }
    
    /* Input field */
    .stTextInput > div, .stNumberInput > div {
        background-color: white !important;
        border-radius: 5px;
    }
    
    /* Judul H1 utama */
    h1 {
        color: white !important;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Select box */
    .stSelectbox > div {
        background-color: rgba(52, 73, 94, 1);
        border-radius: 5px;
    }

    /* Deploy button styling */
    button[kind="primary"] {
        background-color: #3498db !important;
        color: white;
        border-radius: 5px !important;
        border: none !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #2980b9 !important;
    }
    
    /* Alternatif styling untuk deploy button */
    .stDeployButton button {
        background-color: #3498db !important;
        color: white !important;
    }   

     /* Responsif untuk mobile */
    @media (max-width: 768px) {
        /* Memperbaiki ukuran container */
        .header-container, .content-container {
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Memperbaiki ukuran tabel agar bisa di-scroll */
        .stDataFrame {
            width: 100%;
            overflow-x: auto;
        }
        
        /* Memperkecil font untuk mobile */
        h1 {
            font-size: 24px !important;
        }
        
        h2, h3 {
            font-size: 20px !important;
        }
        
        p, label, span {
            font-size: 14px !important;
        }
        
        /* Memperbaiki ukuran peta */
        iframe {
            height: 300px !important;
        }
    }                     
</style>
""", unsafe_allow_html=True)

# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers

    return c * r

# Function to get coordinates
@st.cache_data
def get_coordinates(address):
    """Convert address to latitude and longitude coordinates"""
    try:
        geolocator = Nominatim(user_agent="tourism_recommender_streamlit", timeout=10)
        full_address = f"{address}, Yogyakarta, Indonesia"

        for _ in range(3):  # Maximum 3 attempts
            try:
                time.sleep(1)  # Delay to avoid API rate limiting
                location = geolocator.geocode(full_address)
                if location:
                    return location.latitude, location.longitude
            except GeocoderTimedOut:
                continue

        return None, None

    except Exception as e:
        st.error(f"Error getting coordinates: {str(e)}")
        return None, None

# Tourism Recommendation System Class
class TourismRecommendationSystem:
    def __init__(self, data_path):
        # Load data
        self.data = pd.read_csv(data_path, sep=";")
        self.original_data = self.data.copy()

        # Placeholder for models
        self.knn_model = None
        self.best_k = None
        self.scaler = None
        self.feature_cols = None
        self.normalized_data = None
        self.labels = None

    def prepare_data(self):
        """Prepare data for KNN model training"""
        # Verify required columns
        required_columns = ['Objek Wisata', 'Latitude', 'Longitude', 'HTM Weekday', 'HTM Weekend', 'Rating', 'Grup', 'Kategori Wisata']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Columns {', '.join(missing_columns)} not found in dataset")

        # Handle missing values
        self.data = self.data.dropna()

        # Define features for KNN model
        self.feature_cols = ['HTM Weekday', 'HTM Weekend', 'Rating']

        # Save original features and create ranking
        self.original_features = self.data[self.feature_cols].copy()
        self.ranked_data = self.original_features.rank(method='average')

        # Normalize data
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.ranked_data)
        self.normalized_df = pd.DataFrame(
            self.normalized_data,
            columns=self.feature_cols,
            index=self.ranked_data.index
        )

        # Prepare labels
        self.labels = self.data['Grup'].values
        self.unique_groups = np.unique(self.labels)

    def train_knn_model(self):
        """Train KNN model with fixed parameters: k=9 and test_size=0.2"""
        # Fixed parameters
        self.best_k = 9
        test_size = 0.2
        random_state = 42

        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(
            self.normalized_data, self.labels, test_size=test_size, random_state=random_state
        )

        self.knn_model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.knn_model.fit(X_train, y_train)

        return self.knn_model

    def show_filtered_results(self, user_lat, user_lon, max_distance, max_budget, preferred_categories=None, is_weekend=False):
        """Filter tourism spots based on distance, budget, and categories"""
        # Calculate distance from user location to all tourism spots
        self.data['Jarak'] = self.data.apply(
            lambda row: haversine_distance(user_lat, user_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

        # Determine HTM column to use
        htm_col = 'HTM Weekend' if is_weekend else 'HTM Weekday'

        # Create basic filters
        filter_distance = self.data[self.data['Jarak'] <= max_distance].copy()
        filter_budget = self.data[self.data[htm_col] <= max_budget].copy()

        # Filter by category
        if preferred_categories and len(preferred_categories) > 0:
            filter_category = self.data[self.data['Kategori Wisata'].isin(preferred_categories)].copy()
        else:
            filter_category = self.data.copy()

        # Combined filters: distance + budget
        filter_jarak_budget = self.data[(self.data['Jarak'] <= max_distance) &
                                      (self.data[htm_col] <= max_budget)].copy()

        # Combined all criteria
        if preferred_categories and len(preferred_categories) > 0:
            filter_combined = self.data[(self.data['Jarak'] <= max_distance) &
                                      (self.data[htm_col] <= max_budget) &
                                      (self.data['Kategori Wisata'].isin(preferred_categories))].copy()
        else:
            filter_combined = filter_jarak_budget.copy()

        return filter_distance, filter_budget, filter_category, filter_combined, filter_jarak_budget

    def get_recommendations(self, user_lat, user_lon, max_distance, max_budget, is_weekend=False):
        """Get tourism recommendations using KNN model"""
        # Make sure KNN model is trained
        if self.knn_model is None:
            self.train_knn_model()

        # Calculate distance from user location to all tourism spots
        self.data['Jarak'] = self.data.apply(
            lambda row: haversine_distance(user_lat, user_lon, row['Latitude'], row['Longitude']),
            axis=1
        )

        # Determine HTM column
        htm_col = 'HTM Weekend' if is_weekend else 'HTM Weekday'

        # Create user feature vector
        user_features = np.zeros(self.normalized_data.shape[1])

        # Fill feature vector with appropriate values
        htm_idx = self.feature_cols.index(htm_col)
        rating_idx = self.feature_cols.index('Rating')

        # Normalize budget with Min-Max Scaling
        min_htm = self.data[htm_col].min()
        max_htm = self.data[htm_col].max()
        if max_htm > min_htm:  # To avoid division by zero
            budget_scaled = (max_budget - min_htm) / (max_htm - min_htm)
            user_features[htm_idx] = budget_scaled
        else:
            user_features[htm_idx] = 1.0

        # Normalize rating
        user_features[rating_idx] = self.normalized_df['Rating'].mean()

        # Use KNN model to get nearest neighbors
        distances, neighbor_indices = self.knn_model.kneighbors(
            [user_features], n_neighbors=self.best_k, return_distance=True
        )

        # Flatten arrays
        distances = distances.flatten()
        neighbor_indices = neighbor_indices.flatten()

        # Get neighbor data and add similarity distance
        neighbor_data = self.data.iloc[neighbor_indices].copy()
        neighbor_data['euclidean_distance'] = distances

        euclidean_normalization = pd.DataFrame()
        euclidean_normalization['Objek Wisata'] = neighbor_data['Objek Wisata']
        euclidean_normalization['Raw Euclidean Distance'] = neighbor_data['euclidean_distance']
        euclidean_normalization['Inverted Euclidean'] = neighbor_data['euclidean_distance'].apply(
            lambda x: 1 / (1 + x)  # Direct inversion without normalization
        )
        euclidean_normalization = euclidean_normalization.sort_values(by='Raw Euclidean Distance')
        
        # Identify dominant group from nearest neighbors
        dominant_grup = neighbor_data['Grup'].value_counts().idxmax()
        dominant_grup_count = neighbor_data['Grup'].value_counts().max()

        # Filter data based on criteria
        filter_conditions = [
            self.data['Jarak'] <= max_distance,
            self.data[htm_col] <= max_budget
        ]

        # Combine all filters
        filtered_data = self.data.copy()
        for condition in filter_conditions:
            filtered_data = filtered_data[condition]
            
        # Check if there are any tourism spots that match the criteria
        if filtered_data.empty:
            st.warning("Tidak ditemukan objek wisata yang sesuai dengan kriteria jarak, budget, dan kategori.")
            return pd.DataFrame()

        # Separate by dominant group
        dominant_group_places = filtered_data[filtered_data['Grup'] == dominant_grup].copy()
        other_group_places = filtered_data[filtered_data['Grup'] != dominant_grup].copy()

        # Add flag for dominant group
        dominant_group_places['is_dominant_grup'] = True
        other_group_places['is_dominant_grup'] = False

        # Combine with priority to dominant group
        final_recommendations = pd.concat([dominant_group_places, other_group_places])

        final_recommendations['raw_distance'] = final_recommendations['Jarak']
        final_recommendations['distance_rank'] = final_recommendations['Jarak'].rank(method='min')
    
        max_rank = final_recommendations['distance_rank'].max()
        min_rank = final_recommendations['distance_rank'].min()
        
        if max_rank > min_rank:
            # Invert rank (rank 1/closest = highest value)
            final_recommendations['normalized_distance'] = final_recommendations['distance_rank'].apply(
                lambda r: 1 - ((r - min_rank) / (max_rank - min_rank))
            )
        else:
            # If all ranks are the same
            final_recommendations['normalized_distance'] = 0.5
        
        objek_to_idx = {self.data.loc[idx, 'Objek Wisata']: idx for idx in self.data.index}
        
        final_recommendations['normalized_rating'] = 0.0
        final_recommendations['normalized_htm'] = 0.0

        for idx, row in final_recommendations.iterrows():
            objek_wisata = row['Objek Wisata']
            if objek_wisata in objek_to_idx:
                original_idx = objek_to_idx[objek_wisata]
                # Use pre-normalized rating value
                final_recommendations.loc[idx, 'normalized_rating'] = self.normalized_df.loc[original_idx, 'Rating']
                
                # Use pre-normalized HTM value
                htm_col_norm = 'HTM Weekend' if is_weekend else 'HTM Weekday'
                final_recommendations.loc[idx, 'normalized_htm'] = self.normalized_df.loc[original_idx, htm_col_norm]
    
        # Add column for raw euclidean distance
        final_recommendations['raw_euclidean_distance'] = np.nan
        
        # Set default similarity score
        final_recommendations['normalized_similarity'] = 0.5  # Default value

        for i, row in final_recommendations.iterrows():
            objek_wisata = row['Objek Wisata']
            if objek_wisata in neighbor_data['Objek Wisata'].values:
                # Find index in neighbor_data
                neighbor_idx = neighbor_data[neighbor_data['Objek Wisata'] == objek_wisata].index[0]
                
                # Get euclidean distance
                euclidean_dist = neighbor_data.loc[neighbor_idx, 'euclidean_distance']
                final_recommendations.loc[i, 'raw_euclidean_distance'] = euclidean_dist
                
                # Calculate similarity score with direct inversion
                final_recommendations.loc[i, 'normalized_similarity'] = 1 / (1 + euclidean_dist)

        final_recommendations['dominant_group_score'] = final_recommendations['is_dominant_grup'].astype(int)

        
        #Calculate score components
        final_recommendations['rating_component'] = final_recommendations['normalized_rating'] * 0.2
        final_recommendations['htm_component'] = final_recommendations['normalized_htm'] * 0.2
        final_recommendations['distance_component'] = final_recommendations['normalized_distance'] * 0.25
        final_recommendations['similarity_component'] = final_recommendations['normalized_similarity'] * 0.2
        final_recommendations['group_component'] = final_recommendations['dominant_group_score'] * 0.15
        
        # Calculate total score
        final_recommendations['total_score'] = (
            final_recommendations['rating_component'] +
            final_recommendations['htm_component'] +
            final_recommendations['distance_component'] +
            final_recommendations['similarity_component'] +
            final_recommendations['group_component']
        )

        # Sort by total score
        final_recommendations = final_recommendations.sort_values('total_score', ascending=False)

        # Format recommendation results
        result_columns = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'is_dominant_grup',
                        'Grup', 'Jarak', 'HTM Weekday', 'HTM Weekend', 'Latitude', 'Longitude',
                        'normalized_rating', 'normalized_distance', 'normalized_htm',
                        'normalized_similarity', 'total_score', 'Url']
        
        # Include any columns from the original data that were in the result
        available_columns = [col for col in result_columns if col in final_recommendations.columns]
        result = final_recommendations[available_columns].copy()
    
        return result

# Function to create map with markers
def create_map(user_lat, user_lon, places_df=None, title="Map"):
    # Create map centered on user location
    m = folium.Map(location=[user_lat, user_lon], zoom_start=12)
    
    # Add user marker
    folium.Marker(
        [user_lat, user_lon],
        popup='Your Location',
        tooltip='Your Location',
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)
    
    # Add markers for tourism spots if provided
    if places_df is not None and not places_df.empty:
        for _, place in places_df.iterrows():
            # Customize icon color based on group
            group_colors = {1: 'blue', 2: 'green'}
            color = group_colors.get(place['Grup'], 'cadetblue')
            
            # Create popup content
            popup_content = f"""
            <b>{place['Objek Wisata']}</b><br>
            Kategori: {place['Kategori Wisata']}<br>
            Rating: {place['Rating']}/5<br>
            Jarak: {place['Jarak']:.2f} km<br>
            HTM Weekday: Rp {place['HTM Weekday']}<br>
            HTM Weekend: Rp {place['HTM Weekend']}<br>
            """
            
            folium.Marker(
                [place['Latitude'], place['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=place['Objek Wisata'],
                icon=folium.Icon(color=color)
            ).add_to(m)
    
    return m

# Main Streamlit app
def main():
    # Tambahkan kode deteksi mobile di sini
    detect_mobile = """
    <script>
    // Deteksi apakah pengguna menggunakan perangkat mobile
    if (window.innerWidth < 768) {
        // Tandai bahwa ini adalah perangkat mobile
        localStorage.setItem('is_mobile', 'true');
    } else {
        localStorage.setItem('is_mobile', 'false');
    }
    </script>
    """
    st.markdown(detect_mobile, unsafe_allow_html=True)
    
    # Cek apakah sudah ada state untuk mobile
    if 'mobile_detected' not in st.session_state:
        st.session_state.mobile_detected = False
    
    # App title and description
    st.title("🏞️ Sistem Rekomendasi Wisata Yogyakarta")
    st.markdown("""
    <div style='background-color: rgba(52, 73, 94, 1); padding: 15px; border-radius: 5px;'>
    Temukan objek wisata berdasarkan lokasi dan budget Anda!
    </div>
    """, unsafe_allow_html=True)
    
    # Load and prepare data
    try:
        data_path = "data.csv"  # Change this to the actual path
        recommendation_system = TourismRecommendationSystem(data_path)
        recommendation_system.prepare_data()
        recommendation_system.train_knn_model()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Hasil Rekomendasi", "Tentang Sistem"])
    
    with tab1:        
        with st.sidebar:
            st.header("Preferensi Wisata")
            
            address = st.text_input("Lokasi Anda (contoh: Jalan Malioboro)")
            max_budget = st.sidebar.number_input(
                "Budget Maksimum",
                min_value=0,
                max_value=200000,
                value=50000,
                step=5000
            )
            # Tampilkan nilai terformat
            st.sidebar.write(f"Rp{max_budget:,}".replace(",", "."))
            
            distance_input = st.sidebar.text_input("Jarak Maksimum (km)", "5")
            try:
                max_distance = int(distance_input)
                if max_distance < 0:
                    st.sidebar.warning("Jarak tidak boleh negatif, menggunakan 0 km")
                    max_distance = 0
                elif max_distance > 50:
                    st.sidebar.warning("Jarak maksimal 50 km, menggunakan 50 km")
                    max_distance = 50
            except ValueError:
                st.sidebar.error("Masukkan angka yang valid")
                max_distance = 10
            
            st.markdown("---")  # Separator
            
            time_options = ["Weekday (Senin-Jumat)", "Weekend (Sabtu-Minggu)"]
            visit_time = st.radio("Waktu Kunjungan", time_options)
            is_weekend = (visit_time == time_options[1])
            
            # Get all unique categories
            all_categories = sorted(recommendation_system.data['Kategori Wisata'].unique())
            preferred_categories = st.multiselect("Kategori Wisata", all_categories)

            search_button = st.button("Cari Rekomendasi", type="primary", use_container_width=True)
        
        # Get user coordinates
        if search_button:
            with st.spinner("Mencari rekomendasi objek wisata..."):
                user_lat, user_lon = get_coordinates(address)
                
                if user_lat is None or user_lon is None:
                    st.error("Gagal mendapatkan koordinat lokasi. Silakan periksa alamat Anda.")
                    st.stop()
                
                # Get filter results
                filter_distance, filter_budget, filter_category, filter_combined, filter_jarak_budget = recommendation_system.show_filtered_results(
                    user_lat, user_lon, max_distance, max_budget, preferred_categories, is_weekend
                )
                
                # Get recommendations
                recommendations = recommendation_system.get_recommendations(
                    user_lat, user_lon, max_distance, max_budget, is_weekend
                )
            
            # Display filter results in tab
            filter_tab1, filter_tab2, filter_tab3, rec_tab = st.tabs([
                "Objek Wisata Berdasarkan Jarak & Budget", 
                "Objek Wisata Berdasarkan Kategori", 
                "Objek Wisata Berdasarkan Kombinasi Jarak, Budget, dan Kategori",
                "Rekomendasi"
            ])
            
            with filter_tab1:
                st.markdown("""
                <div style='background-color: rgba(52, 73, 94, 1); padding: 15px; border-radius: 5px;'>
                <h3>Objek Wisata Berdasarkan Jarak dan Budget</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display tables in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Jarak (≤ {max_distance} km)")
                    st.write(f"Ditemukan {len(filter_distance)} tempat wisata.")
                    if not filter_distance.empty:
                        display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', 'HTM Weekday', 'HTM Weekend']
                        df_display = filter_distance[display_cols].sort_values('Jarak').reset_index(drop=True)
                        df_display.index = df_display.index + 1    # Mengubah indeks dimulai dari 1
                        st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                        st.dataframe(df_display, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)                        
                    else:
                        st.info("Tidak ada objek wisata yang memenuhi kriteria jarak.")
                
                with col2:
                    htm_col = 'HTM Weekend' if is_weekend else 'HTM Weekday'
                    st.subheader(f"Budget (≤ Rp{max_budget})")
                    st.write(f"Ditemukan {len(filter_budget)} tempat wisata.")
                    if not filter_budget.empty:
                        display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', htm_col]
                        df_display = filter_budget[display_cols].sort_values(htm_col).reset_index(drop=True)
                        df_display.index = df_display.index + 1
                        st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                        st.dataframe(df_display, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada objek wisata yang memenuhi kriteria budget.")
                
                # Display maps in full width, one by one
                # Map for distance filter
                st.subheader("Peta Objek Wisata (Berdasarkan Jarak)")
                if not filter_distance.empty:
                    map_distance = create_map(user_lat, user_lon, filter_distance, "Objek Wisata Berdasarkan Jarak")
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    folium_static(map_distance, width=800, height=500)
                else:
                    st.info("Tidak ada objek wisata yang memenuhi kriteria jarak.")
                
                # Map for budget filter
                st.subheader("Peta Objek Wisata (Berdasarkan Budget)")
                if not filter_budget.empty:
                    map_budget = create_map(user_lat, user_lon, filter_budget, "Objek Wisata Berdasarkan Budget")
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    folium_static(map_budget, width=800, height=500)
                else:
                    st.info("Tidak ada objek wisata yang memenuhi kriteria budget.")
                
            
            with filter_tab2:
                st.markdown("""
                <div style='background-color: rgba(52, 73, 94, 1); padding: 15px; border-radius: 5px;'>
                <h3>Objek Wisata Berdasarkan Kategori</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if preferred_categories and len(preferred_categories) > 0:
                    st.write(f"Kategori: {', '.join(preferred_categories)}")
                    st.write(f"Ditemukan {len(filter_category)} tempat wisata.")
                    if not filter_category.empty:
                        display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', 'HTM Weekday', 'HTM Weekend']
                        df_display = filter_category[display_cols].sort_values('Rating', ascending=False).reset_index(drop=True)
                        df_display.index = df_display.index + 1  # Mengubah indeks dimulai dari 1
                        st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                        st.dataframe(df_display, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)       
                else:
                    st.info("Tidak ada preferensi kategori yang dipilih.")
                
                st.subheader("Peta Objek Wisata (Berdasarkan Kategori)")
                if not filter_category.empty:
                    map_category = create_map(user_lat, user_lon, filter_category, "Objek Wisata Berdasarkan Kategori")
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    folium_static(map_category, width=800, height=500)
                else:
                    st.info("Tidak ada objek wisata yang memenuhi kriteria kategori.")
            
            with filter_tab3:
                st.markdown("""
                <div style='background-color: rgba(52, 73, 94, 1); padding: 15px; border-radius: 5px;'>
                <h3>Objek Wisata Berdasarkan Kombinasi Kriteria</h3>
                </div>
                """, unsafe_allow_html=True)

                st.subheader(f"Kombinasi Jarak (≤ {max_distance} km) dan Budget (≤ Rp{max_budget})")
                st.write(f"Ditemukan {len(filter_jarak_budget)} tempat wisata.")
                if not filter_jarak_budget.empty:
                    display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', htm_col]
                    df_display = filter_jarak_budget[display_cols].sort_values(['Jarak', htm_col]).reset_index(drop=True)
                    df_display.index = df_display.index + 1
                    st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                    st.dataframe(df_display, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display map for distance + budget filter
                st.subheader("Peta Objek Wisata (Berdasarkan Jarak & Budget)")
                if not filter_jarak_budget.empty:
                    map1 = create_map(user_lat, user_lon, filter_jarak_budget, "Objek Wisata Berdasarkan Jarak & Budget")
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    folium_static(map1, width=800, height=500)
                else:
                    st.info("Tidak ada objek wisata yang memenuhi kriteria jarak dan budget.")
                
                st.subheader(f"Objek Wisata Berdasarkan Kombinasi Jarak (≤ {max_distance} km), Budget (≤ Rp{max_budget}), dan Kategori Wisata")
                st.write(f"Ditemukan {len(filter_combined)} tempat wisata yang memenuhi semua kriteria.")
                if not filter_combined.empty:
                    display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', 'HTM Weekday', 'HTM Weekend']
                    df_display = filter_combined[display_cols].sort_values('Rating', ascending=False).reset_index(drop=True)
                    df_display.index = df_display.index + 1   # This changes the index to start from 1
                    st.markdown('<div style="width:100%; overflow-x:auto;">', unsafe_allow_html=True)
                    st.dataframe(df_display, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Tidak ada objek wisata yang memenuhi semua kriteria. Coba longgarkan kriteria pencarian Anda.")
            
            with rec_tab:
                st.markdown("""
                <div style='background-color: rgba(52, 73, 94, 1); padding: 15px; border-radius: 5px;'>
                <h3>Rekomendasi Wisata Untuk Anda</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if recommendations.empty:
                    st.warning("Tidak ada rekomendasi wisata yang sesuai dengan kriteria Anda. Coba ubah jarak maksimum atau budget Anda.")
                else:
                    dominant_grup = recommendations[recommendations['is_dominant_grup']]['Grup'].iloc[0] if not recommendations[recommendations['is_dominant_grup']].empty else "N/A"
                                    
                    
                    # Format recommendations for display
                    display_cols = ['Objek Wisata', 'Kategori Wisata', 'Rating', 'Jarak', 'HTM Weekday', 'HTM Weekend', 'Url']
                    recommendations_display = recommendations[display_cols].copy()
                    recommendations_display['Jarak'] = recommendations_display['Jarak'].round(2)
                    

                    recommendations_display = recommendations_display.reset_index(drop=True)
                    recommendations_display.index = recommendations_display.index +1

                    recommendations_clickable = recommendations_display.copy()
                    recommendations_clickable['Objek Wisata'] = recommendations_clickable.apply(
                        lambda row: f'<a href="{row["Url"]}" target="_blank">{row["Objek Wisata"]}</a>' if pd.notna(row.get("Url")) else row["Objek Wisata"],
                        axis=1
                    )
                    
                    recommendations_clickable = recommendations_clickable.drop('Url', axis=1)

                    # Simple custom CSS for table
                    css = """
                    <style>
                        /* Target the entire table including all elements */
                        table.dataframe {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        
                        /* Target all header cells and data cells */
                        table.dataframe th, table.dataframe td {
                            padding: 8px;
                            text-align: left;
                            border: 1px solid rgba(0, 0, 0, 0.8);
                            background-color: rgba(0, 0, 0, 0.8);
                        }
                        
                        /* Target header cells specifically */
                        table.dataframe thead th {
                            background-color: rgba(0, 0, 0, 0.8);
                        }
                        
                        /* Target index column specifically */
                        table.dataframe tbody th, table.dataframe thead tr th:first-child {
                            background-color: rgba(0, 0, 0, 0.8);
                            border: 1px solid rgba(0, 0, 0, 0.8);
                        }
                    </style>
                    """

                    # Apply the styling to your existing table HTML output
                    st.markdown(css, unsafe_allow_html=True)
                    
                    # Display top recommendations
                    st.subheader("Top 10 Rekomendasi")
                    st.markdown(f"""
                    <div style="width:100%; overflow-x:auto;">
                        {recommendations_clickable.head(10).to_html(escape=False)}
                    </div>    
                    """, unsafe_allow_html=True)
                    
                    # Display map for recommendations
                    st.subheader("Peta Rekomendasi Wisata")
                    rec_map = create_map(user_lat, user_lon, recommendations.head(10), "Rekomendasi Wisata")
                    if 'mobile_detected' not in st.session_state:
                        st.session_state.mobile_detected = False
                        
                    # Tentukan ukuran peta berdasarkan perangkat
                    map_width = "100%" 
                    map_height = 300 if st.session_state.mobile_detected else 500
                    folium_static(rec_map, width=800, height=500)
                    
    
    with tab2:
        st.markdown("""
        <div style='background-color: rgba(0, 0, 0, 0.8); padding: 15px; border-radius: 5px;'>
        <h3>Tentang Sistem Rekomendasi Wisata</h3>
        
        <p>Sistem rekomendasi wisata ini menggunakan metode K-Nearest Neighbors (KNN) untuk merekomendasikan objek wisata di Daerah Istimewa Yogyakarta
        berdasarkan preferensi pengguna.</p>
        
        <p><b>Cara Kerja Sistem:</b></p>
        <ol>
            <li>Pengguna memasukkan lokasi, jarak maksimum, budget, waktu kunjungan, dan preferensi kategori wisata</li>
            <li>Sistem menghitung jarak dari lokasi pengguna ke seluruh objek wisata</li>
            <li>Sistem menerapkan filter berdasarkan jarak, budget, dan kategori</li>
            <li>Algoritma KNN digunakan untuk menemukan grup wisata yang paling sesuai dengan preferensi pengguna</li>
            <li>Hasil rekomendasi ditampilkan berdasarkan skor total yang dihitung dari berbagai faktor</li>
        </ol>
        
        <p><b>Fitur Utama:</b></p>
        <ul>
            <li>Filter berdasarkan jarak, budget, dan kategori wisata</li>
            <li>Visualisasi lokasi wisata pada peta interaktif</li>
            <li>Peringkat rekomendasi berdasarkan kesesuaian dengan preferensi pengguna</li>
            <li>Detail lengkap tentang objek wisata yang direkomendasikan</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
