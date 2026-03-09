def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Tính toán stride
    stride = image_size / feature_size

    # Khởi tạo list các anchor box
    boxes = []

    # Duyệt theo thứ tự: hàng (i), cột (j), scales, aspect_ratios
    for i in range(feature_size):
        for j in range(feature_size):
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    w = scale * (aspect_ratio ** 0.5)
                    h = scale / (aspect_ratio ** 0.5)
                    
                    # [x_min, y_min, x_max, y_max]
                    box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                    boxes.append(box)
                    
    return boxes