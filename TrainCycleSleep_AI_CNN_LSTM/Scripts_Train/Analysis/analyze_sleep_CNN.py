import glob
import os
import numpy as np
import scipy.signal
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.utils.multiclass import unique_labels

from TrainCNN6lop import (
    hmm_smoothing_viterbi, CONFIG, load_single_subject, SEED, load_trained_model_for_inference
)
from fine_tune_subject_v2_CNN import (
    run_finetuning_for_subject as run_finetuning_for_subject_cnn
)
SLEEP_IMPACT_FACTORS_DETAIL = {
    "thanh_thieu_nien_tre": {
        "min_age": 15, "max_age": 30,
        "factors": [
            {"desc": "Áp lực tâm lý: Stress học tập, công việc, lo âu, rối loạn cảm xúc.", "impacts": ["Wake", "REM", "N3"]},
            {"desc": "Thói quen sinh hoạt: Sử dụng thiết bị điện tử, ánh sáng xanh.", "impacts": ["Wake", "N1", "N2"]},
            {"desc": "Giờ giấc ngủ không đều, thức khuya.", "impacts": ["N3", "REM", "Wake"]},
            {"desc": "Tiêu thụ chất kích thích (caffeine, rượu, thuốc lá).", "impacts": ["Wake", "REM"]},
        ]
    },
    "trung_nien": {
        "min_age": 31, "max_age": 65,
        "factors": [
            {"desc": "Áp lực cuộc sống: Căng thẳng công việc, tài chính, gia đình.", "impacts": ["Wake", "N3"]},
            {"desc": "Các bệnh lý nền: Ngưng thở khi ngủ, Đau mạn tính, Béo phì.", "impacts": ["N3", "Wake"]},
            {"desc": "Thay đổi hormone (nữ giới: tiền mãn kinh, mãn kinh gây bốc hỏa).", "impacts": ["Wake"]},
            {"desc": "Các bệnh lý khác: Cao huyết áp, tiểu đường.", "impacts": ["N3", "Wake"]},
        ]
    },
    "cao_tuoi": {
        "min_age": 66, "max_age": 120,
        "factors": [
            {"desc": "Thay đổi sinh lý tự nhiên: Giảm chất lượng giấc ngủ sâu, giảm melatonin.", "impacts": ["N3", "N2"]},
            {"desc": "Rối loạn nhịp sinh học (thức dậy sớm).", "impacts": ["TotalSleepTime"]},
            {"desc": "Các bệnh lý và thuốc: Tiểu đêm, Ngưng thở khi ngủ, Hội chứng chân không yên.", "impacts": ["Wake"]},
            {"desc": "Đau xương khớp, bệnh tim mạch, Alzheimer.", "impacts": ["N3", "Wake"]},
            {"desc": "Các yếu tố tâm lý - xã hội: Cô đơn, trầm cảm, sự thay đổi lớn (nghỉ hưu).", "impacts": ["Wake", "REM"]},
        ]
    }
}

SLEEP_STAGE_IMPACT_SUMMARY = {
    "Wake": {
        "desc": "Giai đoạn tỉnh táo giữa các chu kỳ ngủ.",
        "function": "Giúp não chuyển giai đoạn, thường ngắn (dưới 8%).",
        "if_high": "Tỉ lệ Wake cao làm giảm chất lượng giấc ngủ, gây mệt mỏi, uể oải.", "if_low": "",
        "improve": "Tránh caffeine/rượu, giảm stress, giữ phòng ngủ tối và yên tĩnh."
    },
    "N1": {
        "desc": "Giai đoạn ngủ nông, dễ bị đánh thức.",
        "function": "Chuyển tiếp từ tỉnh sang ngủ sâu hơn.",
        "if_high": "Nếu quá nhiều N1 → giấc ngủ bị phân mảnh, không phục hồi.", "if_low": "",
        "improve": "Giữ môi trường yên tĩnh, nhiệt độ mát, tránh thức khuya."
    },
    "N2": {
        "desc": "Giai đoạn ngủ vừa – chiếm phần lớn thời gian ngủ.",
        "function": "Củng cố trí nhớ và hồi phục cơ bắp nhẹ.",
        "if_high": "Nếu quá nhiều mà N3/REM thấp → ngủ chưa đủ sâu hoặc do stress.",
        "if_low": "Tỉ lệ N2 thấp bất thường có thể do giấc ngủ bị gián đoạn nhiều.",
        "improve": "Tăng vận động ban ngày, kiểm soát lo âu, duy trì lịch ngủ đều đặn."
    },
    "N3": {
        "desc": "Giấc ngủ sâu, quan trọng cho phục hồi thể chất.",
        "function": "Tăng tiết hormone tăng trưởng, tái tạo mô, tăng miễn dịch.",
        "if_low": "Thiếu N3 → dễ mệt, đau nhức, khó tập trung, suy giảm miễn dịch.", "if_high": "",
        "improve": "Tập thể dục đều đặn, giữ phòng tối, tránh caffeine & rượu bia."
    },
    "REM": {
        "desc": "Giai đoạn mơ, phục hồi não bộ và cảm xúc.",
        "function": "Củng cố trí nhớ, cân bằng cảm xúc, tăng cường sáng tạo.",
        "if_low": "Thiếu REM → khó tập trung, hay quên, dễ cáu gắt, giảm khả năng sáng tạo.", "if_high": "",
        "improve": "Giảm stress, thiền định, ngủ đủ 7–9h, tránh thức khuya."
    }
}
def get_optimal_wakeup_times(sleep_stage_seq, start_time, choice, age, gender):
    optimal_times = []
    if choice == '1':
        for i, stage in enumerate(sleep_stage_seq): # type: ignore
            wakeup_time = start_time + timedelta(seconds=(i + 1) * 30)
            if stage in ['N1', 'N2', 'REM']:
                optimal_times.append(wakeup_time.strftime("%H:%M"))
    elif choice == '2':
        total_minutes = len(sleep_stage_seq) * 0.5 # mỗi sample = 0.5 phút
        num_cycles = int(total_minutes // 90) # type: ignore
        for i in range(1, num_cycles + 1):
            wakeup_time = start_time + timedelta(minutes=90 * i)
            optimal_times.append(wakeup_time.strftime("%H:%M"))
    else:
        print("⚠️ Lựa chọn không hợp lệ. Sử dụng mặc định: 90 phút.")
        return get_optimal_wakeup_times(sleep_stage_seq, start_time, '2', age, gender)

    if choice == '1':
        if gender.lower() == 'nam':
            print("💡 Nam giới thường có ít giấc ngủ REM hơn, cần đảm bảo ngủ sâu.")
        elif gender.lower() == 'nữ':
            print("💡 Nữ giới thường có nhiều REM hơn, quan trọng cho trí nhớ & cảm xúc.")
    elif choice == '2' and age.isdigit() and int(age) > 65:
        print("💡 Người lớn tuổi thường ngủ ngắn hơn, có thể thử dậy sớm hơn.")

    unique_times = []
    if optimal_times:
        unique_times.append(optimal_times[0])
        for t in optimal_times[1:]:
            if t != unique_times[-1]:
                unique_times.append(t)

    return unique_times
def calculate_sleep_quality_score(stage_counts, age):
    """
    Tính toán điểm chất lượng giấc ngủ dựa trên tỉ lệ các giai đoạn.
    """
    total_epochs = sum(stage_counts.values())
    if total_epochs == 0:
        return 0, "Không đủ dữ liệu"

    age_int = int(age) if age.isdigit() else 40

    THRESHOLDS = {
        'N3': {'range': (0.13, 0.23) if age_int > 60 else (0.15, 0.25), 'weight': 40},
        'REM': {'range': (0.18, 0.23) if age_int > 60 else (0.20, 0.25), 'weight': 35},
        'Wake': {'range': (0.02, 0.08), 'weight': 25}
    }

    total_score = 0

    for stage in ['N3', 'REM']:
        percentage = stage_counts.get(stage, 0) / total_epochs
        min_p, max_p = THRESHOLDS[stage]['range']
        weight = THRESHOLDS[stage]['weight']
        
        if percentage >= min_p:
            stage_score = min(1.0, (percentage - min_p) / (max_p - min_p))
        else:
            stage_score = max(0, percentage / min_p)
        total_score += stage_score * weight

    wake_percentage = stage_counts.get('Wake', 0) / total_epochs
    min_p_wake, max_p_wake = THRESHOLDS['Wake']['range']
    if wake_percentage <= max_p_wake:
        wake_score = 1.0
    else:
        wake_score = max(0, 1.0 - (wake_percentage - max_p_wake) / (0.20 - max_p_wake)) # Giảm dần đến 20%
    total_score += wake_score * THRESHOLDS['Wake']['weight']

    final_score = int(np.clip(total_score, 0, 100))
    rating = "Tốt" if final_score >= 75 else "Trung bình" if final_score >= 50 else "Cần cải thiện"
    return final_score, rating

def generate_stage_impact_report(stage_counts, age, stage_summary):
    """
    Tạo bảng phân tích động, cá nhân hóa về ảnh hưởng của từng giai đoạn giấc ngủ.
    """
    total_epochs = sum(stage_counts.values())
    if total_epochs == 0:
        return ["\n⚠️ Không có đủ dữ liệu để tạo bảng ảnh hưởng."]

    lines = ["\n**📋 BẢNG PHÂN TÍCH ẢNH HƯỞNG CÁC GIAI ĐOẠN GIẤC NGỦ (CÁ NHÂN HÓA)**",
             "| Giai đoạn | Vai trò | Trạng thái | Ảnh hưởng tiềm tàng | Gợi ý cải thiện |",
             "| :--- | :--- | :--- | :--- | :--- |"]

    age_int = int(age) if age.isdigit() else 40
    thresholds = {
        'Wake': (0.02, 0.08),
        'N1': (0.02, 0.08),
        'N2': (0.45, 0.55),
        'N3': (0.13, 0.23) if age_int > 60 else (0.15, 0.25),
        'REM': (0.18, 0.23) if age_int > 60 else (0.20, 0.25),
    }

    for stage, (low, high) in thresholds.items():
        pct = stage_counts.get(stage, 0) / total_epochs
        status = "✅ Tốt"
        effect = "Các chỉ số trong ngưỡng khỏe mạnh."

        if pct < low:
            status = f"⬇️ Thấp ({pct:.1%})"
            effect = stage_summary[stage].get("if_low", "Không có ảnh hưởng tiêu cực đáng kể.")
        elif pct > high:
            status = f"⬆️ Cao ({pct:.1%})"
            effect = stage_summary[stage].get("if_high", "Có thể là dấu hiệu của giấc ngủ kém sâu.")

        lines.append(
            f"| **{stage}** | {stage_summary[stage]['function']} | {status} | {effect} | {stage_summary[stage]['improve']} |"
        )

    return lines

def get_personalized_advice(age, gender, stage_counts, sleep_impact_factors_detail, user_factors=None):
    advice = []
    
    total_epochs = sum(stage_counts.values())
    if total_epochs == 0:
        return ["⚠️ Không có đủ dữ liệu để phân tích và đưa ra lời khuyên."]
        
    # Định nghĩa ngưỡng và mô tả
    THRESHOLDS = {
        'N3': {'min_percent': 0.15, 'desc': "Giấc ngủ sâu (N3)", 'emoji': '💤'}, # Cần > 15%
        'REM': {'min_percent': 0.20, 'desc': "Giấc ngủ REM", 'emoji': '🧠'}, # Cần > 20%
        'Wake': {'max_percent': 0.10, 'desc': "Tỉ lệ thức giấc (Wake)", 'emoji': '⚡️'} # Cần < 10%
    }
    
    poor_stages = []
    
    for stage, config in THRESHOLDS.items():
        count = stage_counts.get(stage, 0)
        percentage = count / total_epochs
        
        if 'min_percent' in config and percentage < config['min_percent']:
            poor_stages.append({'stage': stage, 'percent': percentage, 'config': config})
        elif 'max_percent' in config and percentage > config['max_percent']:
            poor_stages.append({'stage': stage, 'percent': percentage, 'config': config})

    if user_factors:
        advice.append("\n--- 🧩 CÁC YẾU TỐ NGOẠI CẢNH GHI NHẬN ---")
        if user_factors.get("stress"):
            advice.append("⚠️ Bạn đang có dấu hiệu căng thẳng. Stress làm tăng thời gian Wake và giảm REM.")
        if user_factors.get("late_night"):
            advice.append("🌙 Thức khuya làm rối loạn nhịp sinh học, giảm giấc ngủ sâu (N3).")
        if user_factors.get("device_usage"):
            advice.append("📱 Sử dụng thiết bị điện tử trước khi ngủ có thể làm giảm chất lượng N2 và REM.")
        if user_factors.get("caffeine"):
            advice.append("☕ Caffeine có thể kéo dài thời gian Wake và giảm REM nếu dùng sau 16h.")
        if user_factors.get("alcohol"):
            advice.append("🍷 Rượu, thuốc lá làm giảm N3 và REM, khiến giấc ngủ không phục hồi.")
        if not user_factors.get("exercise"):
            advice.append("🏃‍♂️ Thiếu vận động làm giảm thời lượng N3. Hãy tập thể dục nhẹ buổi sáng hoặc chiều.")

    age_int = int(age)
    age_group_key = None
    if 15 <= age_int <= 30:
        age_group_key = "thanh_thieu_nien_tre"
    elif 31 <= age_int <= 65:
        age_group_key = "trung_nien"
    elif age_int >= 66:
        age_group_key = "cao_tuoi"

    if not age_group_key:
        advice.append(f"Không xác định được nhóm tuổi cho {age_int}.")
        return advice

    all_potential_factors = sleep_impact_factors_detail[age_group_key]['factors']
    
    if poor_stages:
        advice.append(f"Dựa trên nhóm tuổi ({age_int}), chúng tôi đã **kết hợp** kết quả phân tích giấc ngủ với các yếu tố bên ngoài có khả năng gây ra vấn đề của bạn:")
        
        for p_stage in poor_stages:
            stage_name = p_stage['stage']
            stage_desc = p_stage['config']['desc']
            stage_percent = p_stage['percent'] * 100
            stage_emoji = p_stage['config']['emoji']

            advice.append(f"\n--- {stage_emoji} Vấn đề Chính: {stage_desc} ({stage_percent:.1f}%) ---")

            current_factors = [
                f['desc'] for f in all_potential_factors 
                if stage_name in f['impacts']
            ]
            
            if current_factors:
                advice.append(f"💡 Căn cứ theo nhóm tuổi, **{stage_desc} thấp/cao** có thể liên quan đến các yếu tố tiềm ẩn sau:")
                advice.extend([f"   - {f}" for f in current_factors])

            if stage_name == 'Wake' and user_factors and user_factors.get('device_usage'):
                advice.append("📱 **Kết nối trực tiếp**: Tỉ lệ Wake cao và bạn có dùng điện thoại trước ngủ → khả năng cao ánh sáng xanh đang ảnh hưởng trực tiếp đến giấc ngủ của bạn.")
            if stage_name == 'N3' and user_factors and user_factors.get('late_night'):
                advice.append("🌙 **Kết nối trực tiếp**: Tỉ lệ N3 thấp và bạn có thói quen thức khuya → nên cố định giờ ngủ sớm hơn để cải thiện giấc ngủ sâu.")

        advice.append("\n--- ✅ LỜI KHUYÊN HÀNH ĐỘNG TỔNG QUÁT ---")
        
        if any(p['stage'] == 'N3' for p in poor_stages):
            advice.append("💤 Để tăng cường **Giấc ngủ sâu (N3)**: Tập trung vào thói quen ngủ đều đặn, đảm bảo phòng ngủ tối, mát, yên tĩnh, và tăng cường tập thể dục vào ban ngày.")
        
        if any(p['stage'] == 'REM' for p in poor_stages):
            advice.append("🧠 Để cải thiện **tỉ lệ REM**: Hạn chế tuyệt đối các chất kích thích (rượu, caffeine) 4-6 giờ trước khi ngủ và thực hiện các bài tập thư giãn (thiền, thở sâu) để giảm stress.")
            
        if any(p['stage'] == 'Wake' for p in poor_stages):
            advice.append("⚡️ Để giảm **Tỉ lệ thức giấc (Wake)**: Đánh giá lại việc sử dụng thiết bị điện tử, ánh sáng xanh 1 giờ trước ngủ. Nếu Wake cao và bạn có dấu hiệu ngáy, cần cân nhắc khám bác sĩ chuyên khoa hô hấp.")
            
        if gender.lower() == 'nam':
            advice.append("💡 **Dành cho Nam giới**: Cần đảm bảo chất lượng N3 để phục hồi thể chất tốt nhất.")
        elif gender.lower() == 'nữ':
            advice.append("💡 **Dành cho Nữ giới**: Dễ bị ảnh hưởng bởi stress và thay đổi hormone, cần ưu tiên các kỹ thuật giảm lo âu.")
    else:
        advice.append("🎉 Dữ liệu phân tích cho thấy các chỉ số N3, REM và Wake của bạn đang ở mức lý tưởng. Hãy tiếp tục duy trì thói quen sinh hoạt hiện tại!")
    
    return advice
def generate_sleep_quality_table(stage_counts, sleep_start_time, age):
    """
    (CẢI TIẾN) Tạo bảng tóm tắt chất lượng giấc ngủ với định dạng đẹp hơn
    và đánh giá chi tiết hơn.
    """
    total_epochs = sum(stage_counts.values())
    if total_epochs == 0:
        return ["\n⚠️ Không có đủ dữ liệu để tạo bảng phân tích chất lượng giấc ngủ."]

    age_int = int(age) if age.isdigit() else 40 # Mặc định tuổi trung niên
    
    THRESHOLDS = {
        'Wake': {'range': (0.02, 0.08), 'desc': "Thức giấc", 'role': "Thời gian thức trong đêm."},
        'N1':   {'range': (0.02, 0.08), 'desc': "Ngủ nông", 'role': "Giai đoạn chuyển tiếp, dễ bị đánh thức."},
        'N2':   {'range': (0.45, 0.55), 'desc': "Ngủ vừa", 'role': "Chiếm phần lớn thời gian ngủ, củng cố trí nhớ."},
        'N3':   {'range': (0.13, 0.23) if age_int > 60 else (0.15, 0.25), 'desc': "Ngủ sâu", 'role': "Phục hồi thể chất, tăng trưởng, thải độc não."},
        'REM':  {'range': (0.18, 0.23) if age_int > 60 else (0.20, 0.25), 'desc': "Ngủ mơ", 'role': "Xử lý cảm xúc, sáng tạo, củng cố kỹ năng."}
    }

    table_data = []
    STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']

    for stage in STAGE_ORDER:
        count = stage_counts.get(stage, 0)
        percentage = count / total_epochs

        total_minutes = count * 0.5
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        duration_str = f"{hours}h {minutes}m"

        threshold_info = THRESHOLDS.get(stage)
        assessment_str = "N/A"
        recommendation_str = "N/A"

        if threshold_info:
            min_p, max_p = threshold_info['range']
            recommendation_str = f"{min_p*100:.0f}% - {max_p*100:.0f}%"
            
            if percentage < min_p * 0.8: # Rất thấp
                assessment_str = "Rất thấp ⚠️"
            elif percentage < min_p:
                assessment_str = "Thấp (Cần cải thiện)"
            elif percentage > max_p * 1.2: # Rất cao
                assessment_str = "Rất cao ⚠️"
            elif percentage > max_p:
                assessment_str = "Cao (Bất thường)"
            else:
                assessment_str = "Tốt ✅"

        table_data.append({
            "Giai đoạn": f"{threshold_info['desc']} ({stage})",
            "Mô tả": threshold_info['role'],
            "Thời lượng": duration_str,
            "Tỉ lệ %": f"{percentage*100:.1f}%",
            "Ngưỡng Khuyến nghị": recommendation_str,
            "Đánh giá": assessment_str
        })

    df = pd.DataFrame(table_data)
    report_lines = ["\n**📊 BẢNG TÓM TẮT CHẤT LƯỢNG GIẤC NGỦ**\n"]
    report_lines.append(df.to_markdown(index=False))

    total_sleep_minutes = total_epochs * 0.5
    total_hours = int(total_sleep_minutes // 60)
    total_minutes_rem = int(total_sleep_minutes % 60)
    end_time = sleep_start_time + timedelta(minutes=total_sleep_minutes)

    summary_line = (
        f"\n*Tổng thời gian ghi nhận: **{total_hours} giờ {total_minutes_rem} phút** "
        f"(từ {sleep_start_time.strftime('%H:%M')} đến {end_time.strftime('%H:%M')}).*"
    )
    report_lines.append(summary_line)

    return report_lines

def generate_noise_impact_report(y_true, y_pred, config, subject_id="Unknown", output_dir="final_reports"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n===== 📊 PHÂN TÍCH ẢNH HƯỞNG NHIỄU =====")
    is_clean_mask = (y_true != 5) # Nhãn nhiễu là 5 trong file TrainLSTM6lop.py # type: ignore

    total_samples = len(y_true)
    noise_samples = np.sum(~is_clean_mask)
    print(f"Tổng mẫu: {total_samples}, Nhiễu: {noise_samples} ({noise_samples/total_samples*100:.2f}%)")

    y_true_clean = y_true[is_clean_mask]
    y_pred_clean = y_pred[is_clean_mask]
    f1_clean = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0) # type: ignore
    kappa_clean = cohen_kappa_score(y_true_clean, y_pred_clean)

    f1_full = f1_score(y_true, y_pred, average='macro', zero_division=0) # type: ignore
    kappa_full = cohen_kappa_score(y_true, y_pred)

    print("\n--- So sánh hiệu suất ---")
    print(f"✅ Macro F1 (Sạch): {f1_clean:.4f} | Kappa (Sạch): {kappa_clean:.4f}")
    print(f"🔴 Macro F1 (Đầy đủ): {f1_full:.4f} | Kappa (Đầy đủ): {kappa_full:.4f}")
    print(f"📉 Mức độ ảnh hưởng của nhiễu (F1 giảm): {f1_clean - f1_full:.4f}")

    if noise_samples > 0:
        y_pred_on_noise = y_pred[~is_clean_mask]
        noise_pred_counts = pd.Series(y_pred_on_noise).value_counts().sort_index()
        print("\n📌 Phân bố dự đoán của mô hình trên các mẫu thực sự là nhiễu:")
        for stage, count in noise_pred_counts.items(): # type: ignore
            if stage < len(config.SLEEP_STAGE_LABELS):
                print(f"  - Dự đoán là '{config.SLEEP_STAGE_LABELS[stage]}': {count} mẫu ({count / noise_samples * 100:.2f}%)")

    plt.figure(figsize=(6, 6))
    plt.pie([total_samples - noise_samples, noise_samples],
            labels=["Sạch", "Nhiễu"],
            autopct="%1.1f%%", colors=["#66b3ff", "#ff6666"], startangle=90)
    plt.title(f"Tỉ lệ Sạch vs Nhiễu ({subject_id})")
    plt.savefig(os.path.join(output_dir, f"noise_ratio_{subject_id}.png"), dpi=300)
    plt.close()

    pred_labels = [config.SLEEP_STAGE_LABELS[i] for i in y_pred]
    plt.figure(figsize=(8, 6))
    sns.countplot(x=pred_labels, order=config.SLEEP_STAGE_LABELS, palette="viridis")
    plt.title(f"Phân bố dự đoán ({subject_id})")
    plt.xlabel("Giai đoạn")
    plt.ylabel("Số mẫu")
    plt.savefig(os.path.join(output_dir, f"pred_distribution_{subject_id}.png"), dpi=300)
    plt.close()

def plot_sleep_timeline(y_pred, sleep_start_time, config, subject_id="Unknown", output_dir="final_reports"):
    os.makedirs(output_dir, exist_ok=True)

    epochs = np.arange(len(y_pred))
    times = [sleep_start_time + timedelta(seconds=30 * int(i)) for i in epochs]

    plt.figure(figsize=(14, 5))
    plt.step(times, y_pred, where='post', color='royalblue', linewidth=2)
    plt.yticks(range(len(config.SLEEP_STAGE_LABELS)), config.SLEEP_STAGE_LABELS)
    plt.gca().invert_yaxis() # Đưa Wake lên trên cùng
    plt.xlabel("Thời gian")
    plt.ylabel("Giai đoạn")
    plt.title(f"Timeline giấc ngủ ({subject_id})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    timeline_path = os.path.join(output_dir, f"sleep_timeline_{subject_id}.png")
    plt.savefig(timeline_path, dpi=300)
    plt.close()

    print(f"✅ Timeline giấc ngủ đã lưu: {timeline_path}")

def run_inference_grid_search(model, X_proc, y_true):
    """
    Chạy grid search trên các tham số inference để tìm F1-score macro tốt nhất.
    Tương tự logic trong debug_infer.py.
    """
    best = {"f1": -1}
    temps = [0.8, 1.0, 1.2, 1.5, 1.8] # <-- Dãy Temp MỚI
    trans_diags = [0.8, 0.5, 0.3, 0.1] # <-- Dãy Diag MỚI
    channel_options = [False, True] # False: normal, True: swap
    hmm_options = [True, False] # True: HMM, False: argmax

    print("\n===== 🔍 Bắt đầu Grid Search cấu hình Inference =====")

    for swap in channel_options:
        X_try = X_proc[..., ::-1] if swap else X_proc
        try:
            probs = model.predict(X_try, verbose=0)
        except Exception as e:
            print(f"Lỗi khi dự đoán với swap={swap}: {e}")
            continue

        for temp in temps:
            p_tmp = np.clip(probs, 1e-12, 1.0)**(1.0/float(temp))
            p_tmp = p_tmp / p_tmp.sum(axis=1, keepdims=True)

            for apply_hmm in hmm_options:
                if not apply_hmm:
                    preds = np.argmax(p_tmp, axis=1)
                    td = None # Không có HMM diag
                    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
                    if f1 > best["f1"]:
                        best.update({"f1": f1, "swap": swap, "temp": temp,
                                     "apply_hmm": apply_hmm, "trans_diag": td, "preds": preds})
                    continue

                for td in trans_diags:
                    clean_eval = not np.any(y_true == 5)
                    preds = hmm_smoothing_viterbi(p_tmp, trans_diag=td, clean_eval=clean_eval)
                    f1 = f1_score(y_true, preds, average='macro', zero_division=0)
                    if f1 > best["f1"]:
                        best.update({"f1": f1, "swap": swap, "temp": temp,
                                     "apply_hmm": apply_hmm, "trans_diag": td, "preds": preds})

    print("\n--- Kết quả Grid Search ---")
    if best['f1'] > -1:
        best_config_str = (
            f"F1: {best['f1']:.4f} | Swap: {best['swap']} | Temp: {best['temp']} | "
            f"HMM: {best['apply_hmm']} | Diag: {best['trans_diag']}"
        )
        print(f"✅ Cấu hình tốt nhất: {best_config_str}")
        return best["preds"]
    else:
        print("⚠️ Grid search không tìm thấy cấu hình hợp lệ.")
        return np.argmax(model.predict(X_proc, verbose=0), axis=1)

if __name__ == "__main__":
    print("\n\n===== 💡 Phân tích dữ liệu và đề xuất giờ thức dậy =====")
    
    subject_to_analyze = input("▶️ Nhập tên file dữ liệu sóng (ví dụ: 'SC4581'): ")
    age = input("▶️ Nhập tuổi: ")
    gender = input("▶️ Nhập giới tính (Nam/Nữ): ")

    while True:
        sleep_start_time_str = input("▶️ Nhập giờ đi ngủ (HH:MM, ví dụ: 22:00): ")
        try:
            sleep_start_time = datetime.strptime(sleep_start_time_str, "%H:%M")
            break
        except ValueError:
            print("❌ Sai định dạng, thử lại.")

    print("\n🌙 Một vài câu hỏi nhanh để cá nhân hóa phân tích:")
    stress = input("▶️ Bạn có đang căng thẳng, lo âu hoặc stress không? (y/n): ").lower()
    late_night = input("▶️ Bạn có thường xuyên thức khuya (sau 23h) không? (y/n): ").lower()
    device_usage = input("▶️ Bạn có dùng điện thoại/máy tính trước khi ngủ? (y/n): ").lower()
    caffeine = input("▶️ Bạn có dùng cà phê, trà hoặc chất kích thích buổi chiều/tối không? (y/n): ").lower()
    alcohol = input("▶️ Bạn có sử dụng rượu hoặc thuốc lá không? (y/n): ").lower()
    exercise = input("▶️ Bạn có tập thể dục ít nhất 3 lần/tuần không? (y/n): ").lower()

    user_factors = {
        "stress": stress == "y",
        "late_night": late_night == "y",
        "device_usage": device_usage == "y",
        "caffeine": caffeine == "y",
        "alcohol": alcohol == "y",
        "exercise": exercise == "y"
    }

    best_model_path = None
    base_model_path = open("best_model_path.txt").read().strip()
    base_model_dir = os.path.dirname(base_model_path)
    subject_specific_model_path = os.path.join(base_model_dir, f"fine_tuned_v2_{subject_to_analyze}.keras")

    if os.path.exists(subject_specific_model_path):
        best_model_path = subject_specific_model_path
        print(f"✅ Tìm thấy model đã fine-tune riêng cho subject: {best_model_path}")
    else:
        print(f"ℹ️ Không tìm thấy model riêng cho '{subject_to_analyze}'.")
        do_finetune = input("▶️ Bạn có muốn fine-tune một model mới cho subject này để có kết quả tốt nhất? (y/n): ").lower()
        if do_finetune == 'y':
            print(f"\n===== 🚀 Bắt đầu Fine-tuning cho {subject_to_analyze} từ model '{base_model_path}' =====")
            best_model_path = run_finetuning_for_subject_cnn(subject_to_analyze, base_model_path)
            print(f"===== ✅ Fine-tuning hoàn tất. Model mới: '{best_model_path}' =====\n")

    if not best_model_path:
        print(f"⚠️  CẢNH BÁO: Sử dụng model chung '{base_model_path}' vì không có model riêng hoặc người dùng từ chối fine-tune. Kết quả có thể không tối ưu.")
        best_model_path = base_model_path

    if not best_model_path:
        print("❌ Không thể xác định model để sử dụng. Vui lòng chạy training hoặc fine-tuning trước.")
        exit()

    print(f"✅ Sử dụng model: {best_model_path}")
    model = load_trained_model_for_inference(best_model_path)

    X_raw, y_subject_true = load_single_subject(subject_to_analyze)
    if X_raw is None:
        print(f"❌ Không thể tải dữ liệu cho subject {subject_to_analyze}.")
        exit()

    X_list = []
    for i in range(X_raw.shape[0]):
        x = X_raw[i].astype(np.float32)
        x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_CNN, axis=0).astype(np.float32)
        mean = x_r.mean(axis=0, keepdims=True)
        std = x_r.std(axis=0, keepdims=True) + 1e-8
        X_list.append((x_r - mean) / std)
    X_subject = np.stack(X_list).astype(np.float32)
    y_subject_true = np.array(y_subject_true)

    y_pred_final = run_inference_grid_search(model, X_subject, y_subject_true)

    if y_pred_final is not None and len(y_pred_final) > 0:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(best_model_path)), "final_reports")

        try:
            os.makedirs("debug_plots_CNN", exist_ok=True)
            X = np.array(X_subject)  # ensure ndarray
            n_epochs, n_t, n_ch = X.shape
            ch_means = X.reshape(-1, n_ch).mean(axis=0)
            ch_stds = X.reshape(-1, n_ch).std(axis=0)
            np.save("debug_plots/subject_per_channel_mean.npy", ch_means)
            np.save("debug_plots/subject_per_channel_std.npy", ch_stds)
            print("DEBUG: per-channel mean/std saved:", ch_means, ch_stds)

            from scipy.signal import welch
            sf = 100  # typical sfreq — thay nếu khác (một số file in ra sfreq=100)
            fig, axs = plt.subplots(n_ch, 1, figsize=(8, 2.5 * n_ch))
            for c in range(min(n_ch, 2)): # Giới hạn vẽ 2 kênh để tránh lỗi nếu có nhiều kênh
                f, Pxx = welch(X[0, :, c], fs=sf, nperseg=512)
                axs[c].semilogy(f, Pxx)
                axs[c].set_xlabel("Hz"); axs[c].set_ylabel("PSD")
                axs[c].set_title(f"Subject {subject_to_analyze} PSD epoch 0 ch{c}")
            plt.tight_layout()
            plt.savefig(f"debug_plots/{subject_to_analyze}_epoch0_psd.png", dpi=150)
            plt.close()
            print(f"DEBUG: saved PSD -> debug_plots/{subject_to_analyze}_epoch0_psd.png")

            train_sample_path = "debug_plots/training_epoch_sample.npy"
            if os.path.exists(train_sample_path):
                train_epoch = np.load(train_sample_path)  # expects shape (time, channels)
                fig, axs = plt.subplots(n_ch, 1, figsize=(8, 2.5 * n_ch))
                for c in range(min(n_ch, 2)):
                    f_s, P_s = welch(train_epoch[:, c], fs=sf, nperseg=512)
                    f_x, P_x = welch(X[0, :, c], fs=sf, nperseg=512)
                    axs[c].semilogy(f_s, P_s, label="train", alpha=0.8)
                    axs[c].semilogy(f_x, P_x, label="subject", alpha=0.8)
                    axs[c].legend()
                    axs[c].set_title(f"PSD ch{c} train vs subject")
                plt.tight_layout()
                plt.savefig(f"debug_plots/{subject_to_analyze}_psd_vs_train.png", dpi=150)
                plt.close()
                print("DEBUG: saved PSD comparison with training sample")

        except Exception as _e:
            print("DEBUG: failed saving channel stats/PSD:", _e)

        generate_noise_impact_report(y_subject_true, y_pred_final, CONFIG, subject_id=subject_to_analyze, output_dir=output_dir)

        plot_sleep_timeline(y_pred_final, sleep_start_time, CONFIG, subject_id=subject_to_analyze, output_dir=output_dir)

        try:
            y_pred_stages = [CONFIG.SLEEP_STAGE_LABELS[int(x)] for x in np.array(y_pred_final).astype(int)]
        except Exception:
            y_pred_stages = []

        stage_counts = Counter(y_pred_stages)
        print("\n📊 Phân bố Giai đoạn Giấc ngủ (Số Epoch):", stage_counts)

        sleep_score, sleep_rating = calculate_sleep_quality_score(stage_counts, age)
        print("\n" + "═"*25 + " ĐÁNH GIÁ TỔNG QUAN " + "═"*25)
        print(f"💯 Điểm chất lượng giấc ngủ của bạn: {sleep_score} / 100")
        print(f"⭐ Xếp hạng: {sleep_rating}")
        print("═"*70)

        print("\n" + "═"*20 + " BÁO CÁO PHÂN TÍCH GIẤC NGỦ CHI TIẾT " + "═"*20)

        sleep_quality_report = generate_sleep_quality_table(stage_counts, sleep_start_time, age)
        for line in sleep_quality_report:
            print(line)

        stage_impact_report = generate_stage_impact_report(stage_counts, age, SLEEP_STAGE_IMPACT_SUMMARY)
        for line in stage_impact_report:
            print(line)

        print("\n" + "═"*25 + " ĐỀ XUẤT GIỜ THỨC DẬY " + "═"*25)
        print("Bạn muốn thức dậy theo tiêu chí nào?")
        print("1. Dậy trong giai đoạn nhẹ (N1, N2, REM)")
        print("2. Dậy sau mỗi chu kỳ 90 phút")
        choice = input("▶️ Nhập lựa chọn (1 hoặc 2): ")
 
        print(f"\n📌 Ngủ lúc: {sleep_start_time_str}")
        print(f"👤 Tuổi: {age}, Giới tính: {gender}")
 
        optimal_times = get_optimal_wakeup_times(y_pred_stages, sleep_start_time, choice, age, gender)
        if optimal_times:
            print("\n⏰ Giờ thức dậy tối ưu (chọn chế độ " + choice + "):")
            for i, t in enumerate(optimal_times, 1):
                print(f"   {i}. {t}")
        else:
            print("⚠️ Không có giờ thức dậy tối ưu.")

        final_advice = get_personalized_advice(age, gender, stage_counts, SLEEP_IMPACT_FACTORS_DETAIL, user_factors)
 
        print("\n" + "═"*20 + " LỜI KHUYÊN CÁ NHÂN HÓA & NGUYÊN NHÂN " + "═"*20)
        for i, line in enumerate(final_advice):
            print(line)
        print("═"*70)
    else:
        print(f"❌ Không thể xử lý cho subject {subject_to_analyze}.")
