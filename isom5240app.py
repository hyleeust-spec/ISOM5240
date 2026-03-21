import hashlib
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline

FILE_PATH = "28car_tesla_sold_all_pages.xlsx"


@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")

    raw_columns = df.columns.tolist()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
    )

    column_map = {}
    for col in df.columns:
        if col == "model" or "model" in col:
            column_map[col] = "model"
        elif col == "year" or "year" in col:
            column_map[col] = "year"
        elif col == "pricehkd" or ("price" in col and "hkd" in col):
            column_map[col] = "pricehkd"

    df = df.rename(columns=column_map)

    required_columns = ["model", "year", "pricehkd"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required column(s): {missing_columns}. "
            f"Detected columns: {raw_columns}"
        )

    df["model"] = df["model"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["pricehkd"] = pd.to_numeric(df["pricehkd"], errors="coerce")

    df = df.dropna(subset=["model", "year", "pricehkd"])
    df["year"] = df["year"].astype(int)

    return df


@st.cache_resource
def load_damage_classifier():
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


@st.cache_resource
def load_brand_classifier():
    return pipeline(
        "image-classification",
        model="zjs81/Electric-Car-Brand-Classifier"
    )


@st.cache_resource
def load_tesla_model_classifier():
    return pipeline(
        "image-classification",
        model="dima806/tesla_car_model_image_detection"
    )


def save_uploaded_file_temporarily(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def validate_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return True
    except Exception:
        return False


def get_file_hash(uploaded_file):
    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()
    uploaded_file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()


def check_car_damage(valid_path):
    damage_classifier = load_damage_classifier()
    result = damage_classifier(
        valid_path,
        candidate_labels=["a damaged car", "an undamaged car"]
    )
    top_result = max(result, key=lambda x: x["score"])

    if top_result["label"] == "an undamaged car":
        return "Your car is undamaged"
    return "Your car is damaged"


def car_brand(valid_path):
    brand_classifier = load_brand_classifier()
    car_brand_results = brand_classifier(valid_path)
    detected_car_brand = max(car_brand_results, key=lambda x: x["score"])
    return detected_car_brand


def tesla_model_type(valid_path):
    tesla_model_classifier = load_tesla_model_classifier()
    tesla_model_results = tesla_model_classifier(valid_path)
    detected_tesla_model = max(tesla_model_results, key=lambda x: x["score"])
    return detected_tesla_model["label"], detected_tesla_model


def normalize_detected_model(model_label):
    label = str(model_label).strip().upper()
    label = label.replace("-", "_").replace(" ", "_")

    mapping = {
        "MODEL_3": "Model 3",
        "MODEL3": "Model 3",
        "3": "Model 3",
        "MODEL_E": "Model 3",
        "MODELE": "Model 3",
        "E": "Model 3",
        "MODEL_Y": "Model Y",
        "MODELY": "Model Y",
        "Y": "Model Y",
        "MODEL_S": "Model S",
        "MODELS": "Model S",
        "S": "Model S",
        "MODEL_X": "Model X",
        "MODELX": "Model X",
        "X": "Model X",
    }

    return mapping.get(label, model_label)


def get_available_years(df, model_name):
    matched = df[df["model"].str.upper().str.contains(model_name.upper(), na=False)]
    years = (
        matched["year"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    return years


def get_price_range(df, model_name, year):
    matched_rows = df[
        (df["model"].str.upper().str.contains(model_name.upper(), na=False)) &
        (df["year"] == int(year))
    ]

    if matched_rows.empty:
        return None, None, matched_rows

    min_price = matched_rows["pricehkd"].min()
    max_price = matched_rows["pricehkd"].max()
    return min_price, max_price, matched_rows


def analyze_uploaded_image(uploaded_file, df):
    temp_path = save_uploaded_file_temporarily(uploaded_file)

    damage_result = check_car_damage(temp_path)
    if damage_result == "Your car is damaged":
        return {
            "status": "damaged",
            "damage_result": damage_result
        }

    brand_result = car_brand(temp_path)
    if brand_result["label"] != "Tesla Electric Car":
        return {
            "status": "not_tesla",
            "damage_result": damage_result,
            "brand_result": brand_result
        }

    detected_model_raw, model_info = tesla_model_type(temp_path)
    detected_model = normalize_detected_model(detected_model_raw)
    available_years = get_available_years(df, detected_model)

    return {
        "status": "success",
        "damage_result": damage_result,
        "brand_result": brand_result,
        "detected_model_raw": detected_model_raw,
        "detected_model": detected_model,
        "model_info": model_info,
        "available_years": available_years
    }


def main():
    st.set_page_config(page_title="Tesla Resell Price Finder", layout="wide")
    st.title("Tesla Resell Price Finder")
    st.write("Upload a car image, and the app will automatically detect the Tesla model and show the price range.")

    try:
        df = load_data(FILE_PATH)
    except Exception as e:
        st.error(f"Failed to load Excel data: {e}")
        return

    uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    if not validate_image(uploaded_file):
        st.error("Invalid image file.")
        return

    current_file_hash = get_file_hash(uploaded_file)

    if st.session_state.get("last_file_hash") != current_file_hash:
        with st.spinner("Analyzing image automatically..."):
            analysis_result = analyze_uploaded_image(uploaded_file, df)
            st.session_state["last_file_hash"] = current_file_hash
            st.session_state["analysis_result"] = analysis_result
            st.session_state.pop("selected_year", None)

    result = st.session_state.get("analysis_result")

    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right_col:
        if not result:
            return

        st.subheader("Damage Detection")

        if result["status"] == "damaged":
            st.warning(
                "Your car failed the Damage Detection Validation.\n"
                "This car is damaged. It may not be eligible for resale.",
                icon="⚠️"
            )
            return
        else:
            st.success("Your car pass the Damage Detection Validation successfully", icon="✅")

        st.subheader("Brand Detection")

        if result["status"] == "not_tesla":
            st.warning(
                "Your car failed the Brand Detection Validation. It is not a Tesla car and not eligible for resale.",
                icon="⚠️"
            )
            return
        else:
            st.success(f"Detected brand: {result['brand_result']['label']}", icon="✅")

        st.subheader("Tesla Model Detection")
        st.write(f"Detected model label: {result['detected_model_raw']}")
        st.write(f"Mapped model: {result['detected_model']}")
        st.write(f"Confidence: {result['model_info']['score']:.4f}")

        if not result["available_years"]:
            st.warning(f"No available years found in the resale file for {result['detected_model']}.")
            return

        selected_year = st.selectbox(
            "Select year",
            options=result["available_years"],
            index=None,
            placeholder="Choose a year",
            key="selected_year"
        )

        if selected_year is not None:
            min_price, max_price, matched_rows = get_price_range(df, result["detected_model"], selected_year)

            st.subheader("Resale Price Result")

            if matched_rows.empty:
                st.warning("No matching rows found.")
                return

            st.write(f"Model: {result['detected_model']}")
            st.write(f"Year: {selected_year}")
            st.write(f"Matching records: {len(matched_rows)}")
            st.success(f"Price range: HKD {int(min_price):,} - {int(max_price):,}", icon="✅")


if __name__ == "__main__":
    main()
