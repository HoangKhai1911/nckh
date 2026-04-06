# 🧠 Sleep Stage Classification using AI (CNN, LSTM, Random Forest)

## 📌 Giới thiệu

Dự án này tập trung vào việc **phân loại các giai đoạn giấc ngủ (Sleep Stages)** từ tín hiệu EEG sử dụng các mô hình học máy và học sâu.

Các giai đoạn bao gồm:

* Wake (W)
* N1 (Light Sleep)
* N2
* N3 (Deep Sleep)
* REM

---

## 🎯 Mục tiêu

* Xây dựng hệ thống phân loại sleep stage từ dữ liệu EEG
* So sánh hiệu suất giữa các mô hình:

  * CNN
  * LSTM
  * Random Forest
* Đánh giá bằng các metrics: Accuracy, F1-score, Confusion Matrix

---

## 📂 Dataset

This project uses the **Sleep-EDF Expanded Dataset** provided by PhysioNet.

🔗 Official dataset link:
https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/

---

### 🧠 Dataset Description

The Sleep-EDF Expanded dataset is a widely used benchmark for sleep stage classification research.

* Contains **197 full-night polysomnography (PSG) recordings** ([PhysioNet][1])
* Includes multiple physiological signals:

  * EEG (Fpz-Cz, Pz-Oz)
  * EOG (eye movements)
  * EMG (muscle activity)
  * Event markers ([PhysioNet][1])
* Some recordings also include respiration and body temperature signals ([PhysioNet][1])

Each recording is accompanied by **hypnogram annotations** (sleep stages) manually labeled by experts according to standard sleep scoring guidelines ([PhysioNet][1]).

---

### 💤 Sleep Stages

The dataset provides labeled sleep stages:

* W (Wake)
* N1
* N2
* N3
* REM
* M (Movement)
* ? (Unknown)

---

## 📁 Data Format

* Signal files: `.edf` (European Data Format)
* Annotation files: `.edf+` (Hypnogram)

Each PSG file contains multi-channel physiological signals recorded overnight, typically sampled at **100 Hz for EEG/EOG** ([PhysioNet][2]).

---

### 📌 Notes

* Dataset is **not included** in this repository due to large size
* Users must download data manually from the official source
* Proper citation is required when using this dataset in research

---

### 📚 Citation

If you use this dataset, please cite:

> Kemp et al., *Analysis of a sleep-dependent neuronal feedback loop*, IEEE-BME, 2000.

And the PhysioNet resource accordingly.

---

[1]: https://physionet.org/content/sleep-edfx/?utm_source=chatgpt.com "Sleep-EDF Database Expanded v1.0.0"
[2]: https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/?utm_source=chatgpt.com "Sleep-EDF Database Expanded v1.0.0"


---

## ⚙️ Pipeline

1. **Tiền xử lý dữ liệu**

   * Đọc file `.edf`
   * Chuẩn hóa tín hiệu EEG
   * Chia epoch

2. **Feature Engineering (RF)**

   * Trích xuất đặc trưng từ tín hiệu

3. **Model Training**

   * CNN: học đặc trưng không gian
   * LSTM: học chuỗi thời gian
   * Random Forest: baseline ML

4. **Evaluation**

   * Confusion Matrix
   * Classification Report
   * Cross-validation

---

## 🧠 Mô hình sử dụng

### 🔹 CNN

* Trích xuất đặc trưng từ tín hiệu EEG
* Hiệu quả với dữ liệu dạng sóng

### 🔹 LSTM

* Xử lý dữ liệu chuỗi thời gian
* Nắm bắt sự phụ thuộc theo thời gian

### 🔹 Random Forest

* Baseline model
* So sánh với deep learning

---

## 📊 Kết quả

<img width="805" height="562" alt="image" src="https://github.com/user-attachments/assets/aa13a476-18fb-40b6-94b2-603d079f6d30" />


---

## 📁 Cấu trúc thư mục

```bash
.
├── Baseline_RF/                # Random Forest model
├── TrainCycleSleep_AI_CNN_LSTM/
│   ├── Scripts_Train/          # Training scripts
│   ├── Model_Train/            # Training logic
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Cài đặt

```bash
pip install -r TrainCycleSleep_AI_CNN_LSTM/requirements.txt
```

---

## ▶️ Cách chạy

### Train CNN

```bash
python TrainCycleSleep_AI_CNN_LSTM/Scripts_Train/Training/TrainCNN6lop.py
```

### Train LSTM

```bash
python TrainCycleSleep_AI_CNN_LSTM/Scripts_Train/Training/TrainLSTM6lop.py
```

### Random Forest

```bash
python Baseline_RF/rf_v7.py
```

---

## 📦 Model & Data

Do giới hạn GitHub, dataset và model không được lưu trong repo.

📥 **Tải tại:**

* Dataset + Model: (https://drive.google.com/drive/folders/1L6v5g6bjPmWasZPxk1TjTrGdMc41Wr8U?usp=sharing)

---

## 🖥️ Demo App

👉 Xem demo tại:

> https://github.com/HoangKhai1911/nckh-model-app

---

## 📌 Công nghệ sử dụng

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy, Pandas, Matplotlib

---

## 👨‍💻 Tác giả

* Trần Huỳnh An (Chủ Nhiệm đề tài)
* Hoàng Khải
* Đinh Kỳ Tươi
* Huỳnh Thanh Nhuận
* Trần Thái Thanh

---

## 📄 License

Dự án phục vụ mục đích nghiên cứu và học tập.
