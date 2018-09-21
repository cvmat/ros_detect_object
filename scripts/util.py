import cv2

def visualize_result_onto(cv_image, result, label_format='%(score).2f: %(name)s'):
    for n, region in enumerate(result.regions):
        x0 = region.x_offset
        y0 = region.y_offset
        x1 = region.x_offset + region.width - 1
        y1 = region.y_offset + region.height - 1
        cv2.rectangle(cv_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
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
            (255, 255, 255), cv2.FILLED
        )
        cv2.putText(
            cv_image,
            org = (x0, y0 + size[1]),
            color = (255, 0, 0),
            **text_config
        )
    return cv_image
