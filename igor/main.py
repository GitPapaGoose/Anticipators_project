import streamlit as st
import pandas as pd
import numpy as np
import joblib

CURRENT_YEAR = 2020

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
    'One story': '1Story',
    'One and one-half story: 2nd level finished': '1.5Fin',
    'One and one-half story: 2nd level unfinished': '1.5Unf',
    'Two story': '2Story',
    'Two and one-half story: 2nd level finished': '2.5Fin',
    'Two and one-half story: 2nd level unfinished': '2.5Unf',
    'Split Foyer': 'SFoyer',
    'Split Level': 'SLvl'
}


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

@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

try:
    model = load_model()
except:
    st.error("Ошибка: не найден файл 'house_price_model.pkl'")
    st.stop()

st.title("🏡 Прогноз цены дома (Kaggle House Prices)")
st.markdown("Введите характеристики дома для оценки рыночной стоимости.")

st.header("Основные параметры")
col1, col2 = st.columns(2)

with col1:
    year_built = st.slider("Год постройки", 1870, CURRENT_YEAR, 1980)
    overall_qual = st.slider("Общее качество (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Жилая площадь (кв. футов)", 100, 10000, 1500)
    total_bsmt_sf = st.number_input("Площадь подвала (кв. футов)", 0, 5000, 1000)

with col2:
    # Исправлено: max_value всегда >= min_value
    year_remod = st.slider(
        "Год ремонта",
        min_value=year_built,
        max_value=CURRENT_YEAR if year_built < CURRENT_YEAR else CURRENT_YEAR + 1,
        value=min(year_built + 10, CURRENT_YEAR)
    )
    overall_cond = st.slider("Общее состояние (1-10)", 1, 10, 6)
    lot_area = st.number_input("Площадь участка (кв. футов)", 1000, 200000, 10000)
    garage_area = st.number_input("Площадь гаража (кв. футов)", 0, 2000, 500)

# Выбор района с человекочитаемыми названиями
neighborhood_display = st.selectbox(
    "Район",
    options=list(NEIGHBORHOOD_MAPPING.keys()),
    index=list(NEIGHBORHOOD_MAPPING.keys()).index("College Creek")  # default
)
neighborhood = NEIGHBORHOOD_MAPPING[neighborhood_display]

house_style_display = st.selectbox(
    "Стиль дома",
    options=list(HOUSE_STYLE_MAPPING.keys()),
    index=list(HOUSE_STYLE_MAPPING.keys()).index("Two story")
)
house_style = HOUSE_STYLE_MAPPING[house_style_display]

user_inputs = {
    'YearBuilt': year_built,
    'YearRemodAdd': year_remod if year_remod <= CURRENT_YEAR else CURRENT_YEAR,
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

data = {}
for col in ALL_FEATURES:
    if col in user_inputs:
        data[col] = user_inputs[col]
    else:
        data[col] = DEFAULT_VALUES[col]

input_df = pd.DataFrame([data])

input_df['HouseAge'] = CURRENT_YEAR - input_df['YearBuilt']
input_df['RemodAge'] = CURRENT_YEAR - input_df['YearRemodAdd']
input_df['IsOldNotRemod'] = ((input_df['HouseAge'] > 50) &
                             (input_df['RemodAge'] == input_df['HouseAge'])).astype(int)
input_df['QualCondDiff'] = input_df['OverallQual'] - input_df['OverallCond']
input_df['HasGarage'] = (input_df['GarageArea'] > 0).astype(int)
input_df['HasBsmt'] = (input_df['TotalBsmtSF'] > 0).astype(int)
input_df['LotRatio'] = input_df['LotArea'] / input_df['GrLivArea']
input_df['LotRatio'] = input_df['LotRatio'].replace([np.inf, -np.inf], 0)

if st.button("Рассчитать цену"):
    try:
        log_pred = model.predict(input_df)[0]
        price = np.expm1(log_pred)
        st.success(f"💰 Предсказанная цена: **${price:,.0f}**")
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")