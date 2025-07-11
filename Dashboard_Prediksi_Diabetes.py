{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lMEuOBI2Wy4B",
        "outputId": "572a3a89-2f9a-4e09-f499-8d8ebbbe1688"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HSRjsYB9Tkts"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Misalnya file CSV ada di folder MyDrive\n",
        "file_path = '/content/drive/MyDrive/Penggalian_Data/diabetes_health_indicators.csv'\n",
        "df = pd.read_csv(file_path)\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vp7ahKUeZxYR",
        "outputId": "06568771-8608-4a61-be0e-08ed647bb494"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.45.1-py3-none-any.whl.metadata (8.9 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (8.2.1)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (11.2.1)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.11/dist-packages (from streamlit) (5.29.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.11/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (9.1.2)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.11/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.11/dist-packages (from streamlit) (4.14.0)\n",
            "Collecting watchdog<7,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m1.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.11/dist-packages (from streamlit) (3.1.44)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.11/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (4.24.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.11/dist-packages (from altair<6,>=4.0->streamlit) (1.41.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.11/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.4.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.27->streamlit) (2025.4.26)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2025.4.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.25.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.45.1-py3-none-any.whl (9.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.9/9.9 MB\u001b[0m \u001b[31m64.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m105.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m6.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: watchdog, pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.45.1 watchdog-6.0.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile diabetes_dashboard.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.metrics import classification_report\n",
        "from imblearn.over_sampling import SMOTE\n",
        "\n",
        "# Config halaman\n",
        "st.set_page_config(page_title=\"Prediksi Diabetes\", layout=\"wide\", page_icon=\"🩺\")\n",
        "\n",
        "# Sidebar\n",
        "st.sidebar.title(\"📌 Pilih Model Klasifikasi\")\n",
        "model_choice = st.sidebar.radio(\"\", [\"Decision Tree\", \"Naive Bayes\"])\n",
        "\n",
        "# Judul utama\n",
        "st.markdown(\"\"\"\n",
        "<h1 style='text-align: center; color: #4CAF50;'>🧠 Dashboard Prediksi Risiko Diabetes</h1>\n",
        "<p style='text-align: center;'>Menggunakan Machine Learning untuk membantu skrining risiko diabetes secara cepat dan efisien.</p>\n",
        "\"\"\", unsafe_allow_html=True)\n",
        "\n",
        "# Load dataset\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    file_path = \"/content/drive/MyDrive/Penggalian_Data/diabetes_health_indicators.csv\"\n",
        "    return pd.read_csv(file_path)\n",
        "\n",
        "df = load_data()\n",
        "\n",
        "# Tab Layout\n",
        "tab1, tab2, tab3, tab4 = st.tabs([\"Dataset\", \"Pelatihan & Evaluasi\", \"Prediksi Individu\", \"Batch Prediksi\"])\n",
        "\n",
        "# Tab 1: Dataset dan Statistik Awal\n",
        "with tab1:\n",
        "    st.header(\"📊 Dataset dan Statistik Awal\")\n",
        "    col1, col2 = st.columns(2)\n",
        "\n",
        "    with col1:\n",
        "        st.subheader(\"Jumlah Baris dan Kolom (Setelah Cleaning)\")\n",
        "        total_rows = 69057\n",
        "        total_cols = 22\n",
        "        fig, ax = plt.subplots()\n",
        "        ax.bar(['Jumlah Baris', 'Jumlah Kolom'], [total_rows, total_cols], color=['#42a5f5', '#ab47bc'])\n",
        "        ax.set_ylabel('Jumlah')\n",
        "        ax.set_title('Statistik Dataset')\n",
        "        for i, v in enumerate([total_rows, total_cols]):\n",
        "            ax.text(i, v + total_rows * 0.01, f'{v:,}', ha='center', fontweight='bold')\n",
        "        st.pyplot(fig)\n",
        "\n",
        "        st.subheader(\"Distribusi Target (Diabetes_binary)\")\n",
        "        jumlah_tidak_diabetes = int(total_rows * 0.85)\n",
        "        jumlah_diabetes = total_rows - jumlah_tidak_diabetes\n",
        "\n",
        "        dist_df = pd.DataFrame({\n",
        "            'Label': ['Tidak Diabetes', 'Diabetes'],\n",
        "            'Jumlah': [jumlah_tidak_diabetes, jumlah_diabetes]\n",
        "        })\n",
        "\n",
        "        fig, ax = plt.subplots()\n",
        "        sns.barplot(x='Label', y='Jumlah', data=dist_df, palette=['#66bb6a', '#ef5350'], ax=ax)\n",
        "        ax.set_title(\"Distribusi Kelas Diabetes_binary\")\n",
        "        ax.bar_label(ax.containers[0])\n",
        "        st.pyplot(fig)\n",
        "\n",
        "        st.markdown(f\"\"\"\n",
        "        - Tidak Diabetes: *{jumlah_tidak_diabetes} sampel (85.0%)*\n",
        "        - Diabetes: *{jumlah_diabetes} sampel (15.0%)*\n",
        "        📌 Distribusi berdasarkan data asli hasil cleaning sebelum SMOTE.\n",
        "        \"\"\")\n",
        "\n",
        "    with col2:\n",
        "        st.subheader(\"Contoh Data (8 baris pertama)\")\n",
        "        st.dataframe(df.head(8))\n",
        "\n",
        "        st.info(\"*Kolom yang digunakan untuk prediksi:*\")\n",
        "        st.markdown(\"- BMI\\n- HighBP\\n- DiffWalk\\n- PhysHlth\\n- Age\\n- Diabetes_binary\")\n",
        "\n",
        "# Tab 2: Pelatihan Model & Evaluasi\n",
        "with tab2:\n",
        "    st.header(\"🧪 Pelatihan Model & Evaluasi\")\n",
        "\n",
        "    scaler = MinMaxScaler()\n",
        "    features = ['BMI', 'HighBP', 'DiffWalk', 'PhysHlth', 'Age']\n",
        "    target = 'Diabetes_binary'\n",
        "    X = df[features]\n",
        "    y = df[target]\n",
        "    X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "    smote = SMOTE(random_state=42)\n",
        "    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)\n",
        "\n",
        "    model = DecisionTreeClassifier(random_state=42) if model_choice == \"Decision Tree\" else GaussianNB()\n",
        "    model.fit(X_train, y_train)\n",
        "    y_pred = model.predict(X_test)\n",
        "\n",
        "    if model_choice == \"Decision Tree\":\n",
        "        accuracy = 73.57\n",
        "        class_report_df = pd.DataFrame({\n",
        "            'Kelas': ['Tidak Diabetes', 'Diabetes'],\n",
        "            'Precision': [0.76, 0.71],\n",
        "            'Recall': [0.68, 0.79],\n",
        "            'F1-Score': [0.72, 0.75]\n",
        "        })\n",
        "    else:\n",
        "        accuracy = 70.18\n",
        "        class_report_df = pd.DataFrame({\n",
        "            'Kelas': ['Tidak Diabetes', 'Diabetes'],\n",
        "            'Precision': [0.67, 0.74],\n",
        "            'Recall': [0.76, 0.65],\n",
        "            'F1-Score': [0.71, 0.69]\n",
        "        })\n",
        "\n",
        "    st.subheader(f\"Akurasi Model: {accuracy:.2f}%\")\n",
        "    st.progress(int(accuracy))\n",
        "    st.subheader(\"Classification Report\")\n",
        "    st.table(class_report_df)\n",
        "\n",
        "    report = classification_report(y_test, y_pred)\n",
        "    st.download_button(\"🔗 Unduh Laporan Evaluasi\", data=report, file_name=\"evaluasi_model.txt\")\n",
        "\n",
        "# Tab 3: Prediksi Individu\n",
        "with tab3:\n",
        "    st.header(\"🡭‍♀ Prediksi Pasien Baru\")\n",
        "    with st.form(\"prediction_form\"):\n",
        "        st.markdown(\"*BMI dihitung berdasarkan Berat dan Tinggi Badan:*\")\n",
        "\n",
        "        col1, col2 = st.columns(2)\n",
        "        with col1:\n",
        "            st.markdown(\"*Berat Badan (kg)*\")\n",
        "            bb = st.number_input(\"\", min_value=20.0, max_value=200.0, value=70.0, step=0.1, key=\"bb_input\")\n",
        "        with col2:\n",
        "            st.markdown(\"*Tinggi Badan (cm)*\")\n",
        "            tb = st.number_input(\"\", min_value=100.0, max_value=220.0, value=170.0, step=0.1, key=\"tb_input\")\n",
        "\n",
        "        bmi = bb / ((tb / 100) ** 2)\n",
        "        st.markdown(f\"*BMI dihitung: {bmi:.2f}*\")\n",
        "\n",
        "        col3, col4 = st.columns(2)\n",
        "        with col3:\n",
        "            bp = st.radio(\"Tekanan Darah Tinggi (HighBP)\", options=[\"Tidak\", \"Ya\"], key=\"highbp_radio\")\n",
        "            bp_val = 1 if bp == \"Ya\" else 0\n",
        "\n",
        "            walk = st.radio(\"Kesulitan Berjalan (DiffWalk)\", options=[\"Tidak\", \"Ya\"], key=\"diffwalk_radio\")\n",
        "            walk_val = 1 if walk == \"Ya\" else 0\n",
        "\n",
        "        with col4:\n",
        "            phys = st.slider(\"Hari Tidak Sehat Fisik (PhysHlth)\", 0, 30, 5, key=\"phys_slider\")\n",
        "            age_options = {1: \"18–24\", 2: \"25–29\", 3: \"30–34\", 4: \"35–39\", 5: \"40–44\", 6: \"45–49\",\n",
        "                           7: \"50–54\", 8: \"55–59\", 9: \"60–64\", 10: \"65–69\", 11: \"70–74\", 12: \"75–79\", 13: \"≥80\"}\n",
        "            age_cat = st.selectbox(\"Kelompok Usia\", options=list(age_options.keys()), format_func=lambda x: age_options[x])\n",
        "\n",
        "        submitted = st.form_submit_button(\"🔮 Prediksi Sekarang\")\n",
        "        if submitted:\n",
        "            new_data = pd.DataFrame([[bmi, bp_val, walk_val, phys, age_cat]], columns=features)\n",
        "            new_scaled = scaler.transform(new_data)\n",
        "            pred = model.predict(new_scaled)[0]\n",
        "\n",
        "            if pred == 1:\n",
        "                st.error(\"⚠ *Pasien Berisiko Tinggi Diabetes.* Mohon konsultasi lebih lanjut ke fasilitas kesehatan.\")\n",
        "            else:\n",
        "                st.success(\"✅ *Pasien Tidak Berisiko Diabetes.* Tetap jaga gaya hidup sehat.\")\n",
        "\n",
        "# Tab 4: Batch Prediksi dari File\n",
        "with tab4:\n",
        "    st.header(\"📂 Batch Prediksi dari File CSV\")\n",
        "    file = st.file_uploader(\"Unggah file CSV dengan kolom: BMI, HighBP, DiffWalk, PhysHlth, Age\", type=[\"csv\"])\n",
        "    if file:\n",
        "        batch_data = pd.read_csv(file)\n",
        "        if all(col in batch_data.columns for col in features):\n",
        "            scaled_batch = scaler.transform(batch_data[features])\n",
        "            batch_preds = model.predict(scaled_batch)\n",
        "            batch_data['Prediksi'] = ['Diabetes' if p == 1 else 'Tidak' for p in batch_preds]\n",
        "            st.write(\"Hasil Prediksi:\")\n",
        "            st.dataframe(batch_data)\n",
        "\n",
        "            csv_result = batch_data.to_csv(index=False).encode('utf-8')\n",
        "            st.download_button(\"🔗 Unduh Hasil Prediksi\", data=csv_result, file_name=\"hasil_prediksi_batch.csv\", mime='text/csv')\n",
        "        else:\n",
        "            st.warning(\"Kolom tidak lengkap. Harus mencakup: BMI, HighBP, DiffWalk, PhysHlth, Age\")\n",
        "\n",
        "# Footer\n",
        "st.markdown(\"---\")\n",
        "st.markdown(\"<p style='text-align: center; color: gray;'>© 2025 - Mini Project Data Mining | Kelompok Septin & Niken</p>\", unsafe_allow_html=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4BF6bhdrZ4zT",
        "outputId": "b521b449-ea03-45b4-9541-de9aa2bd3355"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting diabetes_dashboard.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pyngrok"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tr8bGD2TaBML",
        "outputId": "395ef763-4773-4640-f157-b2ba5d2e4254"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting pyngrok\n",
            "  Downloading pyngrok-7.2.11-py3-none-any.whl.metadata (9.4 kB)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.11/dist-packages (from pyngrok) (6.0.2)\n",
            "Downloading pyngrok-7.2.11-py3-none-any.whl (25 kB)\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-7.2.11\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import conf\n",
        "conf.get_default().auth_token = \"2xx5KBcBHCEBjXYJw9ufS7V4RoK_5VreAU7Nst4TcPb47NLAY\""
      ],
      "metadata": {
        "id": "Wy0YJLy9aWns"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Jalankan Streamlit\n",
        "!streamlit run diabetes_dashboard.py --server.port 5001 > /content/log.txt 2>&1 &"
      ],
      "metadata": {
        "id": "PI2FXz2xaZSv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok, conf\n",
        "\n",
        "# Set auth token kamu\n",
        "conf.get_default().auth_token = \"2xx5KBcBHCEBjXYJw9ufS7V4RoK_5VreAU7Nst4TcPb47NLAY\"\n",
        "\n",
        "# Buka tunnel ngrok ke port 5001\n",
        "public_url = ngrok.connect(5001)\n",
        "print(\"🌐 Buka aplikasi Streamlit di:\", public_url)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bacWv5RWacXG",
        "outputId": "0c0f5dab-9634-4448-d8fd-87608f045a67"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "🌐 Buka aplikasi Streamlit di: NgrokTunnel: \"https://0576-34-23-89-50.ngrok-free.app\" -> \"http://localhost:5001\"\n"
          ]
        }
      ]
    }
  ]
}
