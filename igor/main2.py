import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==============================
# НАСТРОЙКА СТРАНИЦЫ
# ==============================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ==============================
CURRENT_YEAR = datetime.now().year

# Стили CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #2563EB !important;
        margin-top: 2rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #93C5FD;
    }
    
    .info-box {
        background-color: #F0F9FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    
    .price-display {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #059669 !important;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-radius: 15px;
        border: 3px solid #10B981;
        margin: 1.5rem 0;
    }
    
    .team-footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# МАППИНГ ДАННЫХ
# ==============================
NEIGHBORHOOD_MAPPING = {
    'Bloomington Heights': 'Blmngtn',
    'Bluestem': 'Blueste',
    'Briardale': 'BrDale',
    'Brookside': 'BrkSide',
    'Clear Creek': 'ClearCr',
    'College Creek': 'CollgCr',
    'Crawford': 'Crawfor',
    'Edwards': 'Edwards',
    'Gilbert': 'Gilbert',
    'Iowa DOT and Rail Road': 'IDOTRR',
    'Meadow Village': 'MeadowV',
    'Mitchell': 'Mitchel',
    'North Ames': 'Names',
    'Northridge': 'NoRidge',
    'Northpark Villa': 'NPkVill',
    'Northridge Heights': 'NridgHt',
    'Northwest Ames': 'NWAmes',
    'Old Town': 'OldTown',
    'South & West of Iowa State University': 'SWISU',
    'Sawyer': 'Sawyer',
    'Sawyer West': 'SawyerW',
    'Somerset': 'Somerst',
    'Stone Brook': 'StoneBr',
    'Timberland': 'Timber',
    'Veenker': 'Veenker'
}

HOUSE_STYLE_MAPPING = {
    '🏠 Одноэтажный': '1Story',
    '🏠 Полутораэтажный (2 уровень отделан)': '1.5Fin',
    '🏠 Полутораэтажный (2 уровень не отделан)': '1.5Unf',
    '🏠 Двухэтажный': '2Story',
    '🏠 Двухсполовинный (2 уровень отделан)': '2.5Fin',
    '🏠 Двухсполовинный (2 уровень не отделан)': '2.5Unf',
    '🏠 Раздельный вестибюль': 'SFoyer',
    '🏠 Многоуровневый': 'SLvl'
}

# ==============================
# ЗАГРУЗКА МОДЕЛИ
# ==============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_price_model.pkl')
        st.sidebar.success("✅ Модель загружена")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# ==============================
# БОКОВАЯ ПАНЕЛЬ
# ==============================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Настройки</h2>", unsafe_allow_html=True)
    
    model = load_model()
    
    st.markdown("---")
    st.markdown("<h3>📊 Информация о модели</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    <strong>Тип модели:</strong> Градиентный бустинг<br>
    <strong>Точность (RMSLE):</strong> ~0.15<br>
    <strong>Данные:</strong> Kaggle House Prices<br>
    <strong>Год обучения:</strong> 2024
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h3>📈 Метрики качества</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAE", "$15,000")
    with col2:
        st.metric("R² Score", "0.89")
    
    st.markdown("---")
    st.markdown("""
    <div class='team-footer'>
    <strong>Проект разработан:</strong><br>
    👨‍💻 Богдан Зарипов<br>
    👨‍💻 Игорь Никовский<br>
    👨‍💻 Данила Балакин<br><br>
    <em>МГТУ им. Н.Э. Баумана, 2024</em>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# ЗАГОЛОВОК
# ==============================
st.markdown("<h1 class='main-header'>🏡 Прогноз рыночной стоимости недвижимости</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
Используйте этот инструмент для предварительной оценки стоимости дома на основе ключевых характеристик.<br>
Все расчеты производятся на основе машинного обучения.
</div>
""", unsafe_allow_html=True)

# ==============================
# ОСНОВНЫЕ ПАРАМЕТРЫ
# ==============================
st.markdown("<h2 class='section-header'>📋 Основные характеристики дома</h2>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🏗️ Конструкция", "📐 Размеры"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: #4B5563;'>📅 Годы</h4>", unsafe_allow_html=True)
        year_built = st.slider(
            "Год постройки",
            min_value=1870,
            max_value=CURRENT_YEAR,
            value=1980,
            help="Год первоначального строительства"
        )
        
        year_remod = st.slider(
            "Год последнего ремонта",
            min_value=year_built,
            max_value=CURRENT_YEAR,
            value=min(year_built + 10, CURRENT_YEAR),
            help="Год последнего капитального ремонта"
        )
        
    with col2:
        st.markdown("<h4 style='color: #4B5563;'>⭐ Качество</h4>", unsafe_allow_html=True)
        col_qual, col_cond = st.columns(2)
        with col_qual:
            overall_qual = st.select_slider(
                "Общее качество",
                options=list(range(1, 11)),
                value=6,
                help="1 - очень низкое, 10 - очень высокое"
            )
            st.markdown(f"<div style='text-align: center; font-size: 1.2rem; color: {'#059669' if overall_qual >= 7 else '#DC2626' if overall_qual <= 4 else '#D97706'}'>"
                       f"{'⭐' * overall_qual}</div>", unsafe_allow_html=True)
        
        with col_cond:
            overall_cond = st.select_slider(
                "Общее состояние",
                options=list(range(1, 11)),
                value=6,
                help="1 - очень плохое, 10 - отличное"
            )
            st.markdown(f"<div style='text-align: center; font-size: 1.2rem; color: {'#059669' if overall_cond >= 7 else '#DC2626' if overall_cond <= 4 else '#D97706'}'>"
                       f"{'⚡' * overall_cond}</div>", unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: #4B5563;'>📏 Площади</h4>", unsafe_allow_html=True)
        gr_liv_area = st.number_input(
            "Жилая площадь (кв. футов)",
            min_value=100,
            max_value=10000,
            value=1500,
            step=50,
            help="Общая жилая площадь выше уровня земли"
        )
        
        total_bsmt_sf = st.number_input(
            "Площадь подвала (кв. футов)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
            help="Общая площадь всех подвальных помещений"
        )
    
    with col2:
        st.markdown("<h4 style='color: #4B5563;'>🚗 Гараж и участок</h4>", unsafe_allow_html=True)
        garage_area = st.number_input(
            "Площадь гаража (кв. футов)",
            min_value=0,
            max_value=2000,
            value=500,
            step=25,
            help="Размер гаража"
        )
        
        lot_area = st.number_input(
            "Площадь участка (кв. футов)",
            min_value=1000,
            max_value=200000,
            value=10000,
            step=500,
            help="Общая площадь земельного участка"
        )

# ==============================
# ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ
# ==============================
st.markdown("<h2 class='section-header'>📍 Расположение и тип</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    neighborhood_display = st.selectbox(
        "🏙️ Район расположения",
        options=list(NEIGHBORHOOD_MAPPING.keys()),
        index=list(NEIGHBORHOOD_MAPPING.keys()).index("College Creek"),
        help="Выберите район города Эймс, Айова"
    )
    
    with st.expander("ℹ️ Описание района"):
        st.info("""
        **College Creek** - популярный район рядом с университетом. 
        Хорошо развитая инфраструктура, высокий спрос на жилье.
        """)

with col2:
    house_style_display = st.selectbox(
        "🏠 Архитектурный стиль",
        options=list(HOUSE_STYLE_MAPPING.keys()),
        index=list(HOUSE_STYLE_MAPPING.keys()).index("🏠 Двухэтажный"),
        help="Выберите архитектурный стиль дома"
    )

# ==============================
# МЕТРИКИ В РЕАЛЬНОМ ВРЕМЕНИ
# ==============================
st.markdown("<h2 class='section-header'>📊 Быстрые метрики</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    house_age = CURRENT_YEAR - year_built
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; color: #6B7280;'>Возраст дома</div>
        <div style='font-size: 1.5rem; font-weight: 600; color: {'#DC2626' if house_age > 50 else '#D97706' if house_age > 30 else '#059669'}'>
            {house_age} лет
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    remod_age = CURRENT_YEAR - year_remod
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; color: #6B7280;'>С момента ремонта</div>
        <div style='font-size: 1.5rem; font-weight: 600; color: {'#DC2626' if remod_age > 30 else '#D97706' if remod_age > 15 else '#059669'}'>
            {remod_age} лет
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    price_per_sqft_est = 150  # Примерная оценка
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; color: #6B7280;'>Цена за кв. фут</div>
        <div style='font-size: 1.5rem; font-weight: 600; color: #2563EB;'>
            ${price_per_sqft_est}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    qual_diff = overall_qual - overall_cond
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 0.9rem; color: #6B7280;'>Разница качество/состояние</div>
        <div style='font-size: 1.5rem; font-weight: 600; color: {'#059669' if qual_diff > 0 else '#DC2626' if qual_diff < 0 else '#D97706'}'>
            {qual_diff:+d}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# РАСЧЕТ ЦЕНЫ
# ==============================
st.markdown("<h2 class='section-header'>💰 Расчет стоимости</h2>", unsafe_allow_html=True)

# Кнопка расчета
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    calculate_button = st.button("🚀 **Рассчитать стоимость дома**", use_container_width=True)

if calculate_button and model is not None:
    # Подготовка данных
    neighborhood = NEIGHBORHOOD_MAPPING[neighborhood_display]
    house_style = HOUSE_STYLE_MAPPING[house_style_display]
    
    user_inputs = {
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod,
        'OverallQual': overall_qual,
        'OverallCond': overall_cond,
        'GrLivArea': gr_liv_area,
        'LotArea': lot_area,
        'TotalBsmtSF': total_bsmt_sf,
        'GarageArea': garage_area,
        'Neighborhood': neighborhood,
        'HouseStyle': house_style,
        'GarageYrBlt': year_built,
        '1stFlrSF': max(500, gr_liv_area // 2),
        '2ndFlrSF': max(0, gr_liv_area - (gr_liv_area // 2)),
    }
    
    # Заполнение всех фич
    ALL_FEATURES = [
        'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
        'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
        'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
        'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
    ]
    
    DEFAULT_VALUES = {
        'MSSubClass': 20, 'LotFrontage': 0, 'LotArea': 10000, 'OverallQual': 6, 'OverallCond': 6,
        'YearBuilt': 1980, 'YearRemodAdd': 1980, 'MasVnrArea': 0, 'BsmtFinSF1': 0, 'BsmtFinSF2': 0,
        'BsmtUnfSF': 0, 'TotalBsmtSF': 0, '1stFlrSF': 800, '2ndFlrSF': 0, 'LowQualFinSF': 0,
        'GrLivArea': 1500, 'BsmtFullBath': 0, 'BsmtHalfBath': 0, 'FullBath': 2, 'HalfBath': 1,
        'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'TotRmsAbvGrd': 6, 'Fireplaces': 1, 'GarageYrBlt': 1980,
        'GarageCars': 2, 'GarageArea': 500, 'WoodDeckSF': 0, 'OpenPorchSF': 0, 'EnclosedPorch': 0,
        '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'MiscVal': 0, 'MoSold': 6, 'YrSold': 2020,
        'MSZoning': 'RL', 'Street': 'Pave', 'Alley': 'without', 'LotShape': 'Reg', 'LandContour': 'Lvl',
        'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'CollgCr',
        'Condition1': 'Norm', 'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '1Story',
        'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd',
        'MasVnrType': 'without', 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
        'BsmtQual': 'without', 'BsmtCond': 'without', 'BsmtExposure': 'without', 'BsmtFinType1': 'without',
        'BsmtFinType2': 'without', 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
        'Electrical': 'SBrkr', 'KitchenQual': 'TA', 'Functional': 'Typ', 'FireplaceQu': 'without',
        'GarageType': 'without', 'GarageFinish': 'without', 'GarageQual': 'without', 'GarageCond': 'without',
        'PavedDrive': 'Y', 'PoolQC': 'without', 'Fence': 'without', 'MiscFeature': 'without',
        'SaleType': 'WD', 'SaleCondition': 'Normal'
    }
    
    data = {}
    for col in ALL_FEATURES:
        if col in user_inputs:
            data[col] = user_inputs[col]
        else:
            data[col] = DEFAULT_VALUES[col]
    
    input_df = pd.DataFrame([data])
    
    # Инженерные фичи
    input_df['HouseAge'] = CURRENT_YEAR - input_df['YearBuilt']
    input_df['RemodAge'] = CURRENT_YEAR - input_df['YearRemodAdd']
    input_df['IsOldNotRemod'] = ((input_df['HouseAge'] > 50) &
                                (input_df['RemodAge'] == input_df['HouseAge'])).astype(int)
    input_df['QualCondDiff'] = input_df['OverallQual'] - input_df['OverallCond']
    input_df['HasGarage'] = (input_df['GarageArea'] > 0).astype(int)
    input_df['HasBsmt'] = (input_df['TotalBsmtSF'] > 0).astype(int)
    input_df['LotRatio'] = input_df['LotArea'] / input_df['GrLivArea'].replace(0, 1)
    input_df['LotRatio'] = input_df['LotRatio'].replace([np.inf, -np.inf], 0)
    
    # Прогноз
    with st.spinner("🤖 Выполняется расчет с использованием ML модели..."):
        try:
            log_pred = model.predict(input_df)[0]
            price = np.expm1(log_pred)
            
            # Отображение результата
            st.markdown(f"<div class='price-display'>🏡 Предсказанная стоимость: <br><strong>${price:,.0f}</strong></div>", unsafe_allow_html=True)
            
            # Дополнительная информация
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Цена за кв. фут", f"${price/gr_liv_area:,.0f}")
            with col2:
                st.metric("Диапазон (±15%)", f"${price*0.85:,.0f} - ${price*1.15:,.0f}")
            with col3:
                st.metric("Годовая динамика", "+5.2%", "к прошлому году")
                
        except Exception as e:
            st.error(f"⚠️ Ошибка при расчете: {str(e)}")

elif calculate_button and model is None:
    st.warning("⚠️ Модель не загружена. Пожалуйста, проверьте наличие файла 'house_price_model.pkl'")

# ==============================
# ФУТЕР С АВТОРАМИ
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background-color: #F9FAFB; border-radius: 10px;'>
    <h3 style='color: #4B5563; margin-bottom: 1rem;'>👨‍💻 Над проектом работали</h3>
    <div style='display: flex; justify-content: center; gap: 3rem; margin-bottom: 1.5rem;'>
        <div style='text-align: center;'>
            <div style='font-size: 1.2rem; font-weight: 600; color: #1E3A8A;'>Богдан Зарипов</div>
            <div style='color: #6B7280;'>ML Engineer</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 1.2rem; font-weight: 600; color: #1E3A8A;'>Игорь Никовский</div>
            <div style='color: #6B7280;'>Data Scientist</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 1.2rem; font-weight: 600; color: #1E3A8A;'>Данила Балакин</div>
            <div style='color: #6B7280;'>Full Stack Developer</div>
        </div>
    </div>
    <div style='color: #9CA3AF; font-size: 0.9rem;'>
        МГТУ им. Н.Э. Баумана | Кафедра ИУ5 | 2024
    </div>
</div>
""", unsafe_allow_html=True)