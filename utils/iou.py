import torch


def compute_iou_tensor(tesnor1, tesnor2):
    '''
    (x1,y1,x2,y2)
    can compute batch data
    inspired by  https://www.jb51.net/article/184542.htm
    '''
    x1_a = tesnor1[:,0]
    y1_a = tesnor1[:,1]
    x2_a = tesnor1[:,2]
    y2_a = tesnor1[:,3]

    x1_b = tesnor2[:,0]
    y1_b = tesnor2[:,1]
    x2_b = tesnor2[:,2]
    y2_b = tesnor2[:,3]

    # computing area of each rectangles
    S_a = (x2_a - x1_a) * (y2_a - y1_a)
    S_b = (x2_b - x1_b) * (y2_b - y1_b)

    # computing the sum_area
    sum_area = S_a + S_b

    left_line = torch.max(x1_a, x1_b)
    right_line = torch.min(x2_a, x2_b)
    top_line = torch.max(y1_a, y1_b)
    bottom_line = torch.min(y2_a, y2_b)


    flag1 = left_line >= right_line

    flag2 = top_line >= bottom_line

    intersect_flag = ~(flag1 + flag2)

    intersect = (right_line - left_line) * (bottom_line - top_line)

    intersect = intersect_flag * intersect

    iou = intersect / (sum_area - intersect)

    return iou