# PHẦN 1: BỐI CẢNH & CÂU CHUYỆN DỮ LIỆU

## Slide 1: Trang bìa (Title Slide)
**[Tiêu đề lớn]:** HỆ THỐNG PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG (ADVANCED FRAUD DETECTION)
**[Tiêu đề phụ]:** Tiếp cận bằng Ensemble Learning, Phân tích hành vi & Tối ưu hóa chi phí
**[Người thực hiện]:** [Tên của bạn/Nhóm]

---

## Slide 2: Vấn đề kinh doanh: "Mò kim đáy bể"
**1. Thực trạng:**
*   Gian lận tài chính gây thiệt hại hàng tỷ USD mỗi năm.
*   Dữ liệu mất cân bằng nghiêm trọng: Chỉ **~0.4%** giao dịch là gian lận (Fraud).

**2. Thách thức đánh đổi (The Trade-off):**
*   **Bỏ sót (False Negative):** Ngân hàng mất tiền bồi thường.
*   **Bắt nhầm (False Positive):** Khóa nhầm thẻ khách VIP ➔ Khách hàng giận dữ, rời bỏ dịch vụ.

**3. Mục tiêu:**
*   Tối đa hóa khả năng phát hiện gian lận.
*   Giảm thiểu tối đa sự phiền toái cho khách hàng thật.

> **🗣 Lời thoại:** *"Thưa thầy cô, bài toán này không chỉ là phân loại 0 và 1. Nó là bài toán tìm kiếm 'chiếc kim' 0.4% dưới đáy bể dữ liệu khổng lồ. Thách thức lớn nhất là làm sao bắt được kẻ gian mà không được khóa nhầm thẻ của khách hàng thật."*

---

## Slide 3: Câu chuyện dữ liệu (Data Storytelling)
*(Chèn biểu đồ phân phối giờ giao dịch và biểu đồ số tiền từ phần EDA)*

**1. Insight "Kẻ trộm đêm khuya":**
*   Giao dịch thường: Tập trung 8h sáng - 10h tối.
*   Giao dịch gian lận: Tăng đột biến lúc **2h - 4h sáng** (Khi chủ thẻ đang ngủ).

**2. Insight "Cú chốt hạ":**
*   Kẻ gian thường thực hiện giao dịch với số tiền lớn đột ngột để tẩu tán hạn mức.
*   **Danh mục rủi ro cao:** Mua sắm trực tuyến (`shopping_net`), Tạp hóa (`grocery_pos`).

**➔ Kết luận:** Gian lận có **mẫu hình (pattern)** cụ thể về hành vi và ngữ cảnh.

---

# PHẦN 2: GIẢI PHÁP KỸ THUẬT (METHODOLOGY)

## Slide 4: Feature Engineering - "Trái tim" của hệ thống
**Vấn đề:** Dữ liệu thô (số tiền, thời gian) không đủ để kết luận.
**Giải pháp: Tạo đặc trưng hành vi (Behavioral Features)**

1.  **`amt_zscore` (Hành vi cá nhân):**
    *   So sánh giao dịch hiện tại với *lịch sử chi tiêu của chính người đó*.
    *   *Ví dụ:* Bình thường tiêu \$50, nay tiêu \$5000 ➔ Bất thường (Z-score cao).
2.  **`distance_km` (Ngữ cảnh địa lý):**
    *   Khoảng cách từ chủ thẻ đến nơi quẹt thẻ.
3.  **`amt_vs_category_mean` (Ngữ cảnh loại hình):**
    *   So sánh với mức trung bình ngành hàng.
    *   *Ví dụ:* \$500 mua Tivi là thường, nhưng \$500 mua cà phê là lừa đảo.

**Kỹ thuật:** Sử dụng **Expanding Window** để tính toán, đảm bảo không rò rỉ dữ liệu tương lai (No Data Leakage).

> **🗣 Lời thoại:** *"Máy tính không hiểu con người, nhưng nó hiểu sự bất thường. Chúng em dạy máy tính so sánh hành vi hiện tại với quá khứ của chính chủ thẻ đó, thay vì dùng một quy tắc cứng nhắc."*

---

## Slide 5: Chiến lược xử lý Imbalance - Tại sao không dùng SMOTE?
**1. Cách tiếp cận truyền thống (SMOTE):**
*   Tạo ra dữ liệu giả (Fake Data) để cân bằng.
*   **Nhược điểm:** Chậm, tốn tài nguyên tính toán, có thể gây nhiễu (Noise).

**2. Cách tiếp cận tối ưu (Cost-Sensitive Learning):**
*   Giữ nguyên dữ liệu gốc.
*   Điều chỉnh **trọng số phạt** (`scale_pos_weight`) trong hàm mất mát.
*   **Ưu điểm:**
    *   Tốc độ xử lý nhanh tuyệt đối.
    *   Phản ánh đúng phân phối thực tế.
    *   Mô hình bị "phạt nặng" hơn nếu bỏ sót 1 giao dịch gian lận.

---

## Slide 6: Kiến trúc Model - Sức mạnh của Ensemble Learning
**Mô hình:** Voting Classifier (Soft Voting).
**Thành phần:** Kết hợp 3 thuật toán Gradient Boosting mạnh nhất hiện nay:

1.  **XGBoost:** Hiệu năng cao, ổn định.
2.  **LightGBM:** Tốc độ huấn luyện cực nhanh trên dữ liệu lớn.
3.  **CatBoost:** Xử lý xuất sắc các biến phân loại (Category, Job).

**Cơ chế:** "Trí tuệ đám đông" - Lấy trung bình xác suất dự đoán của 3 mô hình để đưa ra quyết định cuối cùng ➔ Giảm phương sai, tăng độ tin cậy.

---

# PHẦN 3: KẾT QUẢ & TỐI ƯU HÓA

## Slide 7: Tối ưu ngưỡng quyết định (Threshold Optimization)
*(Chèn biểu đồ Precision-Recall Curve)*

**Vấn đề:** Ngưỡng mặc định 0.5 không tối ưu cho dữ liệu lệch.
**Giải pháp:** Thuật toán quét tìm điểm cắt tối đa hóa **F1-Score**.

**Kết quả:**
*   **Optimal Threshold:** **~0.96** (96%).
*   **Ý nghĩa:** Mô hình cực kỳ tự tin. Chỉ khi xác suất gian lận > 96% thì mới báo động.
*   **Lợi ích:** Loại bỏ hầu hết các cảnh báo giả (False Positives).

> **🗣 Lời thoại:** *"Tại sao ngưỡng lại cao đến 96%? Điều này chứng tỏ Feature Engineering của chúng em rất hiệu quả, giúp mô hình phân tách rạch ròi giữa người thường và kẻ gian. Chúng em thà bỏ sót một chút nghi ngờ nhỏ còn hơn là khóa nhầm thẻ của khách hàng."*

---

## Slide 8: Kết quả thực nghiệm (Model Performance)
*(Chèn ảnh Confusion Matrix và Classification Report)*

*   **Precision (Độ chính xác):** **~93%** ➔ Cứ 100 lần báo động, có 93 lần là gian lận thật.
*   **Recall (Độ nhạy):** **~85-90%** ➔ Bắt được hầu hết các giao dịch gian lận.
*   **AUC-PR:** Đạt mức cao, chứng tỏ mô hình hoạt động tốt trên tập dữ liệu mất cân bằng.

---

## Slide 9: Feature Importance - Mô hình học được gì?
*(Chèn biểu đồ Feature Importance từ code)*

**Top Features:**
1.  `amt` (Số tiền) & `amt_log`.
2.  `category` (Loại hàng hóa).
3.  `amt_vs_category_mean` (Độ lệch chuẩn chi tiêu theo nhóm).
4.  `hour` (Giờ giao dịch).

**Kết luận:** Mô hình hoạt động đúng logic nghiệp vụ, tập trung vào **hành vi bất thường** (số tiền lớn, giờ lạ, sai ngữ cảnh) chứ không học vẹt các thông tin nhiễu.

---

# PHẦN 4: TÁC ĐỘNG & KẾT LUẬN

## Slide 10: Giá trị mang lại cho Doanh nghiệp (Business Impact)
1.  **Bảo vệ tài chính:** Giảm thiểu thất thoát tiền bồi thường nhờ Recall cao.
2.  **Trải nghiệm khách hàng:** Precision cao (~93%) giúp giảm số cuộc gọi xác minh không cần thiết, giữ chân khách hàng.
3.  **Hiệu năng hệ thống:** Sử dụng LightGBM/XGBoost giúp model nhẹ, có thể dự đoán **Real-time** (mili-giây) ngay khi khách quẹt thẻ.

---

## Slide 11: Kết luận & Hướng phát triển
**Tổng kết:**
*   Đã xây dựng thành công hệ thống phát hiện gian lận hiệu quả cao.
*   Giải quyết triệt để vấn đề Imbalance Data bằng Cost-Sensitive Learning.
*   Đảm bảo tính đúng đắn của dữ liệu chuỗi thời gian (No Leakage).

**Hướng phát triển:**
*   Tích hợp Deep Learning (LSTM/RNN) để bắt chuỗi hành vi tuần tự phức tạp hơn.
*   Xây dựng API để triển khai lên môi trường Production.

---

## Slide 12: Q&A (Hỏi đáp)
*   Cảm ơn thầy cô và các bạn đã lắng nghe.
*   *(Chuẩn bị sẵn demo code hoặc mở sẵn notebook để show nếu được hỏi)*