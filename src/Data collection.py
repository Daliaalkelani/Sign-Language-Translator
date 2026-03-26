import os
import cv2
import mediapipe as mp
import sys

# تعطيل رسائل TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# تغيير ترميز الإخراج إلى UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# تهيئة MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)

# تحديد مسار البيانات
DATA_DIR = 'D:\\img'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# حساب عدد الفئات
number_of_classes = 50
dataset_size = 500
print(number_of_classes)

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# التحقق من أن الكاميرا تعمل
if not cap.isOpened():
    print("خطأ: لا يمكن فتح الكاميرا.")
    exit()

for class_name in range(number_of_classes):
    
    class_dir = os.path.join(DATA_DIR, str(class_name))

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_name}')

    done = False
    while True:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        if not ret:
            print("خطأ: لا يمكن قراءة الإطار من الكاميرا.")
            break

        # عرض رسالة "Ready?" على الشاشة
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # انتظر الضغط على زر "q"
        if cv2.waitKey(1) == ord('q'):
            break

    # التقاط الصور
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)

        if not ret:
            print("خطأ: لا يمكن قراءة الإطار من الكاميرا.")
            break

        # حفظ الصورة
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        p = os.path.join(class_dir, f'{counter}.jpg')
        print(p)  # الآن سيتم طباعة المسار بدون خطأ
        print(f'Saved image {counter} for class {class_name}')
        counter += 1

        # عرض الإطار
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

# إغلاق الكاميرا وإنهاء النوافذ
cap.release()
cv2.destroyAllWindows()