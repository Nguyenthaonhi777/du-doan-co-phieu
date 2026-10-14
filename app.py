import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Cấu hình trang
st.set_page_config(
    page_title="Dự Đoán Cổ Phiếu LSTM",
    page_icon="📈",
    layout="centered"
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU (LSTM)")
st.write("Ứng dụng dự đoán giá cổ phiếu sử dụng mạng nơ-ron hồi quy **LSTM**.")

# Khung chọn nhanh trên màn hình chính
col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    ticker = st.text_input("Mã cổ phiếu:", value="VNM.VN")
with col_input2:
    start_date = st.text_input("Ngày bắt đầu:", value="2018-01-01")
with col_input3:
    end_date = st.text_input("Ngày kết thúc:", value="2026-10-14")

# Nút bấm Dự báo nằm ngay chính giữa màn hình
predict_btn = st.button("🚀 BẤM NÚT DỰ BÁO", type="primary", use_container_width=True)

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

if predict_btn:
    with st.spinner("⏳ Đang tải dữ liệu và huấn luyện mô hình LSTM... Vui lòng chờ khoảng 20 - 40 giây..."):
        # FIX SEED (ổn định kết quả)
        seed = 42
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        # Tải dữ liệu
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty or len(df) < 100:
            st.error("Không tìm thấy dữ liệu hoặc dữ liệu quá ngắn! Vui lòng kiểm tra lại mã cổ phiếu.")
            st.stop()

        data = df[['Close']].values

        # Chia tập dữ liệu 80% Train - 20% Test
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Tạo chuỗi thời gian 60 ngày
        X_train, y_train = create_sequences(train_scaled, window_size=60)
        test_combined = np.vstack((train_scaled[-60:], test_scaled))
        X_test, y_test = create_sequences(test_combined, window_size=60)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Chia tập Validation (10% của tập Train)
        val_size = int(0.1 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        # Xây dựng mô hình LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Huấn luyện mô hình
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        # Dự báo trên tập Test
        test_pred = model.predict(X_test, verbose=0)
        test_pred = scaler.inverse_transform(test_pred)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Tính các chỉ số đánh giá
        rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
        mae = mean_absolute_error(y_test_real, test_pred)
        r2 = r2_score(y_test_real, test_pred)

        # Dự báo 10 ngày tới
        last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
        last_60_days = last_60_days.reshape(1, 60, 1)
        future_predictions = []

        for i in range(10):
            next_pred = model.predict(last_60_days, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_60_days = np.append(
                last_60_days[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)

    # --- HIỂN THỊ KẾT QUẢ ---
    st.success("✅ Dự báo thành công!")

    # 1. Hiển thị RMSE, MAE, R2
    st.subheader("1. Chỉ số đánh giá mô hình")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("MAE", f"{mae:,.2f}")
    c3.metric("R2 Score", f"{r2:.4f}")

    # 2. Biểu đồ Loss Train vs Validation
    st.subheader("2. Biểu đồ Loss Train vs Validation")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.legend()
    ax1.set_title("Loss Train vs Validation")
    st.pyplot(fig1)

    # 3. Biểu đồ Giá thật vs Dự báo
    st.subheader("3. Biểu đồ Giá thật và Dự báo (Tập Test)")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test_real, label="Giá thật")
    ax2.plot(test_pred, label="Dự báo")
    ax2.legend()
    st.pyplot(fig2)

    # 4. Kết quả dự báo 10 ngày tới
    st.subheader("4. Dự báo 10 ngày tới")
    df_future = pd.DataFrame({
        "Ngày": [f"Ngày {i+1}" for i in range(10)],
        "Giá dự báo": future_predictions.flatten()
    })
    st.dataframe(df_future.style.format({"Giá dự báo": "{:,.2f}"}), use_container_width=True, hide_index=True)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(y_test_real, label="Giá thật")
    ax3.plot(test_pred, label="Dự báo test")
    future_x = range(len(y_test_real), len(y_test_real) + 10)
    ax3.plot(future_x, future_predictions, marker='o', label="10 ngày tới")
    ax3.legend()
    st.pyplot(fig3)
