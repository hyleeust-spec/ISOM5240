import hashlib
import re
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline

# Path to the resale dataset used for year and price lookup.
FILE_PATH = "28car_tesla_sold_all_pages.xlsx"


@st.cache_data
def load_data(file_path):
    # Load Excel data and keep the original column names for error reporting.
    df = pd.read_excel(file_path, engine="openpyxl")

    raw_columns = df.columns.tolist()

    # Normalize column names to improve matching across slightly different formats.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
    )

    # Map possible column name variations to the expected standard names.
    column_map = {}
    for col in df.columns:
        if col == "model" or "model" in col:
            column_map[col] = "model"
        elif col == "year" or "year" in col:
            column_map[col] = "year"
        elif col == "pricehkd" or ("price" in col and "hkd" in col):
            column_map[col] = "pricehkd"

    df = df.rename(columns=column_map)

    # Ensure the minimum required fields exist before continuing.
    required_columns = ["model", "year", "pricehkd"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required column(s): {missing_columns}. "
            f"Detected columns: {raw_columns}"
        )

    # Clean and convert key columns into consistent types for filtering and pricing.
    df["model"] = df["model"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["pricehkd"] = pd.to_numeric(df["pricehkd"], errors="coerce")

    # Drop incomplete rows that cannot be used in estimation.
    df = df.dropna(subset=["model", "year", "pricehkd"])
    df["year"] = df["year"].astype(int)

    return df


@st.cache_resource
def load_damage_classifier():
    # Load the zero-shot model used to determine whether the car is damaged.
    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )


@st.cache_resource
def load_brand_classifier():
    # Load the brand classifier used to detect whether the uploaded car is a Tesla.
    return pipeline(
        "image-classification",
        model="chanc031965/Tesla_Detection"
    )


@st.cache_resource
def load_tesla_model_classifier():
    # Load the Tesla model classifier for Model S, 3, X, and Y recognition.
    return pipeline(
        "image-classification",
        model="dima806/tesla_car_model_image_detection"
    )


def save_uploaded_file_temporarily(uploaded_file):
    # Save the uploaded image to a temporary file because the classifiers expect a file path.
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def validate_image(uploaded_file):
    # Verify that the uploaded file is a readable image before analysis.
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return True
    except Exception:
        return False


def get_file_hash(uploaded_file):
    # Generate a stable hash so the app can detect whether a new image was uploaded.
    uploaded_file.seek(0)
    file_bytes = uploaded_file.getvalue()
    uploaded_file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()


def is_valid_email(email):
    # Basic email format validation for the resale application form.
    pattern = r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"
    return bool(re.match(pattern, email.strip()))


def check_car_damage(valid_path):
    # Classify the uploaded car image as damaged or undamaged.
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
    # Detect the most likely car brand from the uploaded image.
    brand_classifier = load_brand_classifier()
    car_brand_results = brand_classifier(valid_path)
    detected_car_brand = max(car_brand_results, key=lambda x: x["score"])
    return detected_car_brand


def is_tesla_label(label):
    # Normalize label formatting before checking whether the detected brand is Tesla.
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized == "tesla"


def tesla_model_type(valid_path):
    # Detect the most likely Tesla model from the uploaded image.
    tesla_model_classifier = load_tesla_model_classifier()
    tesla_model_results = tesla_model_classifier(valid_path)
    detected_tesla_model = max(tesla_model_results, key=lambda x: x["score"])
    return detected_tesla_model["label"], detected_tesla_model


def normalize_detected_model(model_label):
    # Convert different model label formats into a consistent display/value format.
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
    # Find all available manufacturing years for the detected Tesla model.
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


def get_trimmed_price_range(df, model_name, year):
    # Filter rows by model and year to estimate a realistic resale range.
    matched_rows = df[
        (df["model"].str.upper().str.contains(model_name.upper(), na=False)) &
        (df["year"] == int(year))
    ].copy()

    if matched_rows.empty:
        return None, None, matched_rows

    # Use the interquartile range boundaries to reduce the effect of outlier prices.
    q1 = matched_rows["pricehkd"].quantile(0.25)
    q3 = matched_rows["pricehkd"].quantile(0.75)

    trimmed_rows = matched_rows[
        (matched_rows["pricehkd"] >= q1) &
        (matched_rows["pricehkd"] <= q3)
    ]

    if trimmed_rows.empty:
        return None, None, trimmed_rows

    # Return the minimum and maximum price from the trimmed set as the estimate.
    min_price = trimmed_rows["pricehkd"].min()
    max_price = trimmed_rows["pricehkd"].max()

    return min_price, max_price, trimmed_rows


def analyze_uploaded_image(uploaded_file, df):
    # Run the full analysis pipeline: damage check, brand check, and Tesla model detection.
    temp_path = save_uploaded_file_temporarily(uploaded_file)

    damage_result = check_car_damage(temp_path)
    if damage_result == "Your car is damaged":
        return {
            "status": "damaged",
            "damage_result": damage_result
        }

    brand_result = car_brand(temp_path)
    if not is_tesla_label(brand_result["label"]):
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
    # Configure the Streamlit page and display the main app introduction.
    st.set_page_config(page_title="Tesla Resale Price Estimator", layout="wide")
    st.title("Tesla Resale Price Estimator")
    st.write("Upload a photo of your car — we’ll instantly identify the Tesla model and show you the estimated price range!")
    st.write("")
    st.success("""
    Tesla Resell Program Policy:
    1. Only cars in good condition are eligible for our resale program.
    2. Only Tesla cars qualify for our resale program.
    3. Eligible Tesla models for our resale program include: Model S, Model 3, Model X, and Model Y.
    4. The price shown is an initial estimation only. For more details or to receive a final resale offer, please contact our team.
    """)

    # Load the resale dataset once and stop the app early if loading fails.
    try:
        df = load_data(FILE_PATH)
    except Exception as e:
        st.error(f"Failed to load Excel data: {e}")
        return

    # Accept image uploads in common photo formats.
    uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    # Reject invalid or corrupted image files before model inference.
    if not validate_image(uploaded_file):
        st.error("Invalid image file.")
        return

    # Hash the file so analysis only reruns when the uploaded image changes.
    current_file_hash = get_file_hash(uploaded_file)

    if st.session_state.get("last_file_hash") != current_file_hash:
        with st.spinner("Analyzing image automatically..."):
            analysis_result = analyze_uploaded_image(uploaded_file, df)
            st.session_state["last_file_hash"] = current_file_hash
            st.session_state["analysis_result"] = analysis_result
            st.session_state.pop("selected_year", None)

    result = st.session_state.get("analysis_result")

    # Create a two-column layout for the uploaded image and analysis results.
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right_col:
        if not result:
            return

        st.subheader("Damage Detection")

        # Stop the workflow immediately if the car is detected as damaged.
        if result["status"] == "damaged":
            st.warning(
                "We regret to inform you that your car did not pass the damage detection check. It is not eligible for resale through our program.",
                icon="⚠️"
            )
            return
        else:
            st.success(
                "Perfect - Your car is in good condition with no damage and is eligible for resale!",
                icon="✅"
            )

        st.subheader("Brand Detection")

        # Stop the workflow if the detected car brand is not Tesla.
        if result["status"] == "not_tesla":
            st.warning(
                "We regret to inform you that your car did not pass the brand detection check, as it does not appear to be a Tesla model. It is not eligible for resale through our program.",
                icon="⚠️"
            )
            return
        else:
            st.success(
                "Great news - Your Tesla is eligible for resale!",
                icon="✅"
            )

        st.subheader("Tesla Model Detection")
        st.success(
            f"Nice - Your Tesla is a **{result['detected_model']}**!",
            icon="✅"
        )

        # If the detected model has no matching year data, estimation cannot continue.
        if not result["available_years"]:
            st.warning(f"No available years found in the resale file for {result['detected_model']}.")
            return

        # Let the user choose the manufacturing year for a more precise price estimate.
        selected_year = st.selectbox(
            "Please select the manufacturing year of your Tesla Car.",
            options=result["available_years"],
            index=None,
            placeholder="Choose a year",
            key="selected_year"
        )

        if selected_year is not None:
            min_price, max_price, matched_rows = get_trimmed_price_range(
                df, result["detected_model"], selected_year
            )

            st.subheader("Resale Price Estimation")

            # Warn the user if no usable pricing records remain after trimming outliers.
            if matched_rows.empty:
                st.warning("No matching rows found after removing the lowest and highest quartiles.")
                return

            st.success(
                f"Estimated price range: HKD {int(min_price):,} - {int(max_price):,}",
                icon="✅"
            )

            st.subheader("Resale Application")

            # Collect an email so the team can follow up on the resale application.
            with st.form("resale_application_form"):
                applicant_email = st.text_input(
                    "Leave your email",
                    placeholder="Enter your email address"
                )
                submitted = st.form_submit_button("Submit Application")

            if submitted:
                # Validate that the user entered a non-empty and properly formatted email.
                if not applicant_email.strip():
                    st.error("Please enter your email address.")
                elif not is_valid_email(applicant_email):
                    st.error("Please enter a valid email address.")
                else:
                    st.success(
                        f"Application submitted successfully! We will contact you at {applicant_email} soon."
                    )


if __name__ == "__main__":
    main()
