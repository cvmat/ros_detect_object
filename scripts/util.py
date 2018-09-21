import cv2
import numpy as np

def assign_color(index):
    s_division = 2
    h_division = 36

    base_number = (1 + index) * 123457 # mutiply a prime number for shuffling.
    h_index = base_number % h_division
    s_index = (base_number / h_division) % s_division

    s = np.uint8((s_index + 2) * 255.0 / (s_division + 1))
    h = np.uint8((h_index + float(s_index) / s_division) * 180.0 / h_division)
    v = 255
    bgr = cv2.cvtColor(np.uint8([[[h,s,v]]]), cv2.COLOR_HSV2BGR)[0][0]
    col = tuple([int(n) for n in bgr])
    return col

def visualize_result_onto(cv_image, result, label_format='%(score).2f: %(name)s'):
    for n, region in enumerate(result.regions):
        frame_col = assign_color(result.labels[n])
        label_background_col = frame_col
        label_foreground_col = tuple([255 - c for c in frame_col])
        x0 = region.x_offset
        y0 = region.y_offset
        x1 = region.x_offset + region.width - 1
        y1 = region.y_offset + region.height - 1
        cv2.rectangle(cv_image, (x0, y0), (x1, y1), frame_col, 2)
        label_text = label_format % {
            'score': result.scores[n], 'name': result.names[n]
        }
        text_config = {
            'text': label_text,
            'fontFace': cv2.FONT_HERSHEY_PLAIN,
            'fontScale': 1,
            'thickness': 1,
        }
        size, baseline = cv2.getTextSize(**text_config)
        cv2.rectangle(
            cv_image, (x0, y0), (x0 + size[0], y0 + size[1]),
            label_background_col, cv2.FILLED
        )
        cv2.putText(
            cv_image,
            org = (x0, y0 + size[1]),
            color = label_foreground_col,
            **text_config
        )
    return cv_image
