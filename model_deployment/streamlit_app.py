
"""
Streamlit Deployment Application
--------------------------------
This application performs the following steps:
1. Loads the trained wellness tourism prediction model.
2. Provides an interactive user interface for customer details.
3. Accepts customer demographic and interaction inputs.
4. Generates purchase predictions using the trained model.
5. Displays prediction results and purchase probability.
"""


# Import required libraries
import os
import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------#
# Step 1: Configure Streamlit Page
# --------------------------------------------------#

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------#
# Step 2: Apply Custom CSS Styling
# --------------------------------------------------#

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    h1 {
        font-size: 28px !important;
        margin-top: 0rem !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }

    h2 {
        font-size: 22px !important;
    }

    label {
        font-size: 13px !important;
    }

    header {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        height: 0rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------#
# Step 3: Load Trained Model
# --------------------------------------------------#

@st.cache_resource(show_spinner="Loading model...")
def load_model():

    current_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(
        current_directory
    )

    model_path = os.path.join(
        project_root,
        "trained_models",
        "wellness_model.joblib"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)



try:

    wellness_model = load_model()

except Exception as error:

    st.error(
        f"Unable to load model.\n\n{error}"
    )

    st.stop()



# --------------------------------------------------#
# Step 4: Display Application Banner
# --------------------------------------------------#

banner_path = "media/wellness_tourism_banner.png"

if os.path.exists(banner_path):

    st.image(
        banner_path,
        use_container_width=True
    )

else:

    st.warning(
        "Banner image not found."
    )



# --------------------------------------------------#
# Step 5: Application Header
# --------------------------------------------------#

st.title(
    "Wellness Tourism Package Prediction"
)

st.caption(
    "Predict whether a customer is likely to purchase the Wellness Tourism Package based on customer profile and interaction details."
)



# --------------------------------------------------#
# Step 6: Customer Input Layout
# --------------------------------------------------#

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)

# --------------------------------------------------#
# Step 7: Customer Details Input
# --------------------------------------------------#

with left_col:

    st.subheader(
        "Customer Details"
    )


    c1, c2 = st.columns(
        2,
        gap="small"
    )


    with c1:

        age = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=61,
            value=30
        )


        city_tier = st.selectbox(
            "City Tier (1-Highest Development, 3-Lowest)",
            [1, 2, 3]
        )


        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )


        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=5,
            value=2
        )


    with c2:

        type_of_contact = st.selectbox(
            "Type of Contact",
            [
                "Company Invited",
                "Self Enquiry"
            ]
        )


        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )


        preferred_property_star = st.selectbox(
            "Preferred Property Rating (Stars)",
            [3, 4, 5]
        )


        number_of_trips = st.number_input(
            "Number of Trips (Annual Average)",
            min_value=1,
            max_value=22,
            value=2
        )


    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced",
            "Unmarried"
        ]
    )



# --------------------------------------------------#
# Step 8: Customer Interaction Input
# --------------------------------------------------#

with right_col:

    st.subheader(
        "Customer Interaction Details"
    )


    c4, c5, c6 = st.columns(
        3,
        gap="small"
    )


    with c4:

        passport = st.selectbox(
            "Passport Availability (0-No, 1-Yes)",
            [0, 1]
        )


        own_car = st.selectbox(
            "Own Car Ownership (0-No, 1-Yes)",
            [0, 1]
        )


        number_of_children = st.number_input(
            "Number of Children Visiting (Below Age 5)",
            min_value=0,
            max_value=3,
            value=0
        )


    with c5:

        designation = st.selectbox(
            "Customer Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )


        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            value=30000,
            step=1000
        )


        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King"
            ]
        )


    with c6:

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=1,
            max_value=6,
            value=3
        )


        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score (1-Low, 5-High)",
            [1, 2, 3, 4, 5]
        )


        duration_of_pitch = st.number_input(
            "Duration of Pitch (Minutes)",
            min_value=5,
            max_value=127,
            value=20
        )

# --------------------------------------------------#
# Step 9: Generate Prediction
# --------------------------------------------------#

st.divider()

st.subheader(
    "Prediction Result"
)


predict_button = st.button(
    "Predict Purchase",
    use_container_width=True
)



if predict_button:


    input_data = pd.DataFrame({

        "Age": [age],

        "TypeofContact": [type_of_contact],

        "CityTier": [city_tier],

        "DurationOfPitch": [duration_of_pitch],

        "Occupation": [occupation],

        "Gender": [gender],

        "NumberOfPersonVisiting": [
            number_of_person_visiting
        ],

        "NumberOfFollowups": [
            number_of_followups
        ],

        "ProductPitched": [
            product_pitched
        ],

        "PreferredPropertyStar": [
            preferred_property_star
        ],

        "MaritalStatus": [
            marital_status
        ],

        "NumberOfTrips": [
            number_of_trips
        ],

        "Passport": [
            passport
        ],

        "PitchSatisfactionScore": [
            pitch_satisfaction_score
        ],

        "OwnCar": [
            own_car
        ],

        "NumberOfChildrenVisiting": [
            number_of_children
        ],

        "Designation": [
            designation
        ],

        "MonthlyIncome": [
            monthly_income
        ]

    })


    prediction = wellness_model.predict(
        input_data
    )[0]



    # Display prediction result

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the Wellness Tourism Package."
        )

    else:

        st.error(
            "Customer is unlikely to purchase the Wellness Tourism Package."
        )



    # Display purchase probability

    if hasattr(
        wellness_model,
        "predict_proba"
    ):

        probability = wellness_model.predict_proba(
            input_data
        )[0][1]


        col_a, col_b = st.columns(
            2
        )


        with col_a:

            st.metric(
                "Purchase Probability",
                f"{probability:.2%}"
            )


        with col_b:

            st.progress(
                float(probability)
            )


        st.caption(
            "Probability indicates the likelihood of the customer purchasing the Wellness Tourism Package."
        )
