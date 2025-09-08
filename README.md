# 🔬 Breast Cancer Prediction Web App

## 📌 Giới thiệu
Đây là ứng dụng web đơn giản sử dụng **Machine Learning (Random Forest)** để dự đoán xem khối u vú là **Lành tính (Benign)** hay **Ác tính (Malignant)** dựa trên 5 thuộc tính quan trọng nhất trong bộ dữ liệu ung thư vú (Breast Cancer Dataset - Kaggle).  

Ứng dụng được xây dựng bằng **Flask** với giao diện thân thiện (Bootstrap Template).

---

## 🚀 Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python  
- **Framework Web**: Flask  
- **Machine Learning**: Random Forest Classifier (Scikit-learn)  
- **Thư viện hỗ trợ**: Pandas, Numpy, Bootstrap (giao diện)  

---

## 🧮 Thuật toán
Mô hình sử dụng **Random Forest** để huấn luyện trên tập dữ liệu ung thư vú.  
Chỉ chọn **5 thuộc tính quan trọng nhất** để tăng tốc độ tính toán và dễ triển khai:  
- `area_worst`  
- `concave points_worst`  
- `concave points_mean`  
- `radius_worst`  
- `concavity_mean`  

Độ chính xác của mô hình đạt ~**95-97%** trên tập test.

---

## 🖥️ Giao diện cơ bản

### Trang chính:
- Form nhập dữ liệu gồm 5 thuộc tính quan trọng nhất.
- Nút **Dự đoán**.
- Hiển thị kết quả: **Benign / Malignant**.
- Hiển thị độ chính xác của mô hình.

#### 🖼️ Ví dụ giao diện:
<img width="1859" height="912" alt="image" src="https://github.com/user-attachments/assets/ccf2b087-8595-400a-a95c-23303ab8dd4a" />
<img width="1863" height="910" alt="image" src="https://github.com/user-attachments/assets/7f0f5401-c9aa-49d6-b817-9a254e5a3d58" />


  
