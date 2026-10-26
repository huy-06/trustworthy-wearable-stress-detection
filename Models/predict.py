"""
MODULE DỰ ĐOÁN TRIỂN KHAI (PRODUCTION INFERENCE ENGINE)
Dự án: Nhận diện Căng thẳng Đáng tin cậy từ Cảm biến Cổ tay (WESAD)
Tác giả: Nhóm nghiên cứu DSR301m - Đại học FPT
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class TrustworthyStressPredictor:
    def __init__(self, model_path=None, config_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "trustworthy_xgb_pipeline.joblib")
        if config_path is None:
            config_path = os.path.join(base_dir, "config_inference.json")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
            
        # Nạp pipeline
        pipeline_data = joblib.load(model_path)
        self.scaler = pipeline_data["scaler"]
        self.model = pipeline_data["model"]
        self.features = pipeline_data["features"]
        self.tau = pipeline_data.get("tau", 0.80)
        self.theta_motion = pipeline_data.get("theta_motion", 15.385)
        self.model_name = pipeline_data.get("model_name", "Trustworthy Stress Predictor")
        
        # Nạp config (nếu có)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def predict_single_window(self, feature_input):
        """
        Dự đoán cho một cửa sổ 60s đơn lẻ.
        feature_input: dict (key=tên đặc trưng) hoặc list/array 25 phần tử.
        """
        if isinstance(feature_input, dict):
            # Sắp xếp đúng thứ tự 25 đặc trưng
            feature_vector = np.array([feature_input[col] for col in self.features]).reshape(1, -1)
            var_acc = feature_input.get("acc_mag_var", 0.0)
        elif isinstance(feature_input, (list, np.ndarray)):
            feature_vector = np.array(feature_input).reshape(1, -1)
            # Lấy vị trí cột acc_mag_var
            var_idx = self.features.index("acc_mag_var")
            var_acc = feature_vector[0, var_idx]
        else:
            raise TypeError("feature_input phải là dict, list hoặc numpy.ndarray")

        # 1. Chuẩn hóa đặc trưng
        scaled_vector = self.scaler.transform(feature_vector)
        
        # 2. Dự đoán xác suất đã hiệu chuẩn Isotonic
        prob_stress = float(self.model.predict_proba(scaled_vector)[0, 1])
        prob_non_stress = 1.0 - prob_stress
        confidence = max(prob_stress, prob_non_stress)
        
        # 3. Kiểm tra điều kiện vận động
        is_high_motion = bool(var_acc >= self.theta_motion)
        
        # 4. Áp dụng luật quyết định Chọn lọc (Selective Decision Rule)
        if prob_stress >= self.tau:
            decision = "STRESS"
            status_code = 1
            is_rejected = False
            explanation = f"Mô hình tự tin phát hiện trạng thái Căng thẳng (Độ tin cậy: {prob_stress*100:.1f}% >= ngưỡng an toàn {self.tau*100:.0f}%)."
        elif prob_non_stress >= self.tau:
            decision = "NON_STRESS"
            status_code = 0
            is_rejected = False
            explanation = f"Mô hình tự tin xác nhận Không có Căng thẳng (Độ tin cậy: {prob_non_stress*100:.1f}% >= ngưỡng an toàn {self.tau*100:.0f}%)."
        else:
            decision = "UNCERTAIN"
            status_code = -1
            is_rejected = True
            if is_high_motion:
                explanation = f"TỪ CHỐI DỰ ĐOÁN (Abstention): Người dùng đang vận động cơ học mạnh (Var(ACC)={var_acc:.1f} >= {self.theta_motion:.1f}) và độ tin cậy xác suất chỉ đạt {confidence*100:.1f}% (< {self.tau*100:.0f}%). Tránh phát sinh báo động giả."
            else:
                explanation = f"TỪ CHỐI DỰ ĐOÁN (Abstention): Tín hiệu sinh lý lấp lửng, độ bất định vượt ngưỡng an toàn (Độ tự tin: {confidence*100:.1f}% < {self.tau*100:.0f}%)."

        return {
            "decision": decision,
            "status_code": status_code,
            "is_rejected": is_rejected,
            "confidence": round(confidence, 4),
            "prob_stress": round(prob_stress, 4),
            "prob_non_stress": round(prob_non_stress, 4),
            "is_high_motion": is_high_motion,
            "acc_mag_var": round(float(var_acc), 3),
            "tau_threshold": self.tau,
            "explanation": explanation
        }

    def predict_batch(self, df_input):
        """
        Dự đoán hàng loạt cho DataFrame gồm nhiều cửa sổ quan sát.
        """
        X_mat = df_input[self.features].values
        var_acc = df_input["acc_mag_var"].values
        
        X_scaled = self.scaler.transform(X_mat)
        probs_stress = self.model.predict_proba(X_scaled)[:, 1]
        probs_non_stress = 1.0 - probs_stress
        
        results = []
        for i in range(len(df_input)):
            p1 = probs_stress[i]
            p0 = probs_non_stress[i]
            conf = max(p1, p0)
            hm = bool(var_acc[i] >= self.theta_motion)
            
            if p1 >= self.tau:
                dec = "STRESS"
                code = 1
                rej = False
            elif p0 >= self.tau:
                dec = "NON_STRESS"
                code = 0
                rej = False
            else:
                dec = "UNCERTAIN"
                code = -1
                rej = True
                
            results.append({
                "decision": dec,
                "status_code": code,
                "is_rejected": rej,
                "prob_stress": round(float(p1), 4),
                "confidence": round(float(conf), 4),
                "is_high_motion": hm
            })
            
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== TEST DEMO INFERENCE ENGINE ===")
    predictor = TrustworthyStressPredictor()
    print(f"[*] Đã nạp thành công mô hình: {predictor.model_name}")
    print(f"[*] Số lượng đặc trưng: {len(predictor.features)}")
    print(f"[*] Ngưỡng tin cậy tau = {predictor.tau}, Ngưỡng vận động = {predictor.theta_motion}")
    
    # Test thử 1 mẫu ngẫu nhiên từ dataset
    data_csv = r"h:\My Drive\FPTU\S5\DSR301m\Project\Dataset\wesad_wrist_features.csv"
    if os.path.exists(data_csv):
        df = pd.read_csv(data_csv)
        sample = df.iloc[10].to_dict()
        res = predictor.predict_single_window(sample)
        print("\n[*] Kết quả dự đoán cho mẫu thử nghiệm (Subject:", sample['subject_id'], "Label gốc:", sample['label'], "):")
        for k, v in res.items():
            print(f"    - {k}: {v}")
