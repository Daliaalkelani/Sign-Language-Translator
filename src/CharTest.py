import pickle

import cv2
import mediapipe as mp
import numpy as np

from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display

def draw_arabic_text(frame, text, position, font_path='arial.ttf', font_size=32, color=(255, 0, 0)):
    reshaped_text = arabic_reshaper.reshape(text)  # إعادة تشكيل الأحرف
    bidi_text = get_display(reshaped_text)  # جعل الكتابة من اليمين لليسار

    # تحويل إطار OpenCV إلى صورة PIL
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, bidi_text, font=font, fill=color)

    # تحويل الصورة مرة أخرى إلى OpenCV
    return np.array(img_pil)

model_dict = pickle.load(open('D:\\img_char\\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'ا', 1: 'ب', 2: 'ت',
    3: 'ث', 4: 'ج', 5: 'ح',
    6: 'خ', 7: 'د', 8: 'ذ',
    9: 'ر', 10: 'ز', 11: 'س',
    12: 'ش', 13: 'ص', 14: ' ض',
    15: ' ط', 16: 'ظ', 17: 'ع ',
    18: 'غ', 19: 'ف', 20: ' ق',
    21: ' ك  ', 22: 'ل', 23: ' م',
    24: '  ن ', 25: 'ه', 26: ' و',
    27: ' ي ', 28: 'ال', 29: 'ة ',           
               }
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    if not ret:
      break
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))



        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        frame = draw_arabic_text(frame, predicted_character, (50, 50), font_path="arial.ttf", font_size=40)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()