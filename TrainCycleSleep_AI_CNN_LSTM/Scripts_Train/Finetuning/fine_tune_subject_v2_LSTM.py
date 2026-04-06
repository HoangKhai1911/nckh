import sys, os, numpy as np, scipy.signal, tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import shuffle, class_weight 
from TrainLSTM6lop import (load_trained_model_for_inference, load_single_subject,
                           CONFIG, augment_signal, focal_loss, SEED)
from collections import Counter

def run_finetuning_for_subject(sub, base_model_path):
    """
    Hàm để fine-tune một model cho một subject cụ thể.
    Trả về đường dẫn của model đã được fine-tune.
    """
    model = load_trained_model_for_inference(base_model_path)

    X_raw, y = load_single_subject(sub)
    if X_raw is None:
        print(f"❌ Không thể tải dữ liệu cho subject {sub}.")
        return None

    MIN_EPOCHS_FOR_FINETUNE = 200 
    if X_raw.shape[0] < MIN_EPOCHS_FOR_FINETUNE:
        print(f"⚠️ Dữ liệu của subject {sub} quá ít ({X_raw.shape[0]} epochs). Bỏ qua fine-tuning để tránh làm giảm chất lượng model.")
        return base_model_path 

    X_list=[]
    for i in range(X_raw.shape[0]):
        x = X_raw[i].astype(np.float32)
        x_r = scipy.signal.resample(x, CONFIG.TARGET_LENGTH_LSTM, axis=0).astype(np.float32)
        mean = x_r.mean(axis=0, keepdims=True); std = x_r.std(axis=0, keepdims=True)+1e-8
        X_list.append((x_r-mean)/std)
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y).astype(int)

    cnt = Counter(y.tolist())
    print("Before oversample counts:", cnt)
    target_min = max( int(np.percentile(list(cnt.values()), 50)), 30 ) 
    X_aug, y_aug = [], []
    
    unique_classes, counts = zip(*cnt.items())
    median_count = np.median(counts)

    for cls in unique_classes:
        idxs = np.where(y==cls)[0]
        num_samples = len(idxs)
        reps = int(np.ceil(median_count / num_samples)) if num_samples < median_count and num_samples > 0 else 1

        for r in range(reps):
            for i in idxs:
                xsel = X[i].copy()
                if r>0:
                    xsel = augment_signal(xsel)
                X_aug.append(xsel)
                y_aug.append(cls)
    X_aug = np.stack(X_aug).astype(np.float32)
    y_aug = np.array(y_aug).astype(int)
    print("After oversample counts:", Counter(y_aug.tolist()))

    X_aug, y_aug = shuffle(X_aug, y_aug, random_state=SEED)
    print("✅ Đã xáo trộn dữ liệu fine-tuning.")

    classes = np.unique(y) 
    weights = class_weight.compute_class_weight(
        'balanced', classes=classes, y=y 
    )
    class_weights_dict = dict(zip(classes, weights))

    if 5 in class_weights_dict:
        if (y == 5).sum() / len(y) < 0.01:
            class_weights_dict[5] = min(class_weights_dict[5], 1.0) 
            
    print("Class Weights cho fine-tuning:", class_weights_dict)

    n_out = model.output_shape[-1]
    y_cat = tf.keras.utils.to_categorical(y_aug, num_classes=n_out)

    for layer in model.layers:
        layer.trainable = True

    opt = tf.keras.optimizers.Adam(learning_rate=2e-5) 
    model.compile(optimizer=opt, loss=focal_loss(gamma=2.0), metrics=['accuracy'])

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        X_aug, y_cat, epochs=50, batch_size=16, 
        validation_split=0.1, callbacks=cb, verbose=1,
        class_weight=class_weights_dict 
    )

    out_path = f"fine_tuned_v2_{sub}.keras"
    model.save(out_path)
    print("Saved", out_path)

    preds = np.argmax(model.predict(X), axis=1)
    print("Macro F1 after fine-tune v2:", f1_score(y, preds, average='macro', zero_division=0))
    print(classification_report(
        y, preds,
        labels=list(range(n_out)),
        target_names=CONFIG.SLEEP_STAGE_LABELS[:n_out],
        zero_division=0
    ))
    
    return out_path

if __name__ == "__main__":
    sub_main = sys.argv[1] if len(sys.argv)>1 else input("subject: ")
    model_path_main = open("best_model_path.txt").read().strip()
    run_finetuning_for_subject(sub_main, model_path_main)