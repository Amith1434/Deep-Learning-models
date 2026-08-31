def calculate_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])

    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])

    intersection_width = max(0,x2 - x1)
    intersection_height = max(0,y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )
    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )
    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )
    union = area1 + area2 - intersection
    return intersection / union


box1 = [0,0,100,100]
box2 = [50,50,150,150]
iou = calculate_iou(box1,box2)
print(iou)