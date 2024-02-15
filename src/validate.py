import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def validate(L, res=100):
    assert (
        len(L) == 5
    ), "L must be an array of 6 elements: l1, l2, l3, l4, EE_x and EE_y"
    L = np.array(L)
    l1 = 1.0
    l2 = L[0]
    l3 = L[1]
    l4 = L[2]
    EE = L[-2:]
    EE1 = EE[0]
    EE2 = EE[1]

    res = res
    th1 = np.linspace(0, 2 * np.pi, res)
    unit = np.ones(res)
    delta = np.zeros(res)
    link2_x = l2 * np.cos(th1)
    link2_y = l2 * np.sin(th1)

    # th2 구하는 과정 제2 코사인 법칙
    link_2_4 = np.sqrt(unit * l1**2 + unit * l2**2 - 2 * l1 * l2 * np.cos(th1))

    rel_th2_1 = np.ones(res)
    rel_th2_2 = np.ones(res)

    for i in range(res):
        if th1[i] <= np.pi:
            rel_th2_1[i] = np.arccos(
                np.round(
                    (link_2_4[i] ** 2 + unit[i] * l2**2 - unit[i] * l1**2)
                    / (2.0 * (unit[i] * l2) * (unit[i] * link_2_4[i])),
                    6,
                )
            )
            rel_th2_2[i] = np.arccos(
                np.round(
                    (link_2_4[i] ** 2 + unit[i] * l3**2 - unit[i] * l4**2)
                    / (2.0 * (unit[i] * l3) * (unit[i] * link_2_4[i])),
                    6,
                )
            )
        else:
            rel_th2_1[i] = -np.arccos(
                np.round(
                    (link_2_4[i] ** 2 + unit[i] * l2**2 - unit[i] * l1**2)
                    / (2.0 * (unit[i] * l2) * (unit[i] * link_2_4[i])),
                    6,
                )
            )
            rel_th2_2[i] = np.arccos(
                np.round(
                    (link_2_4[i] ** 2 + unit[i] * l3**2 - unit[i] * l4**2)
                    / (2.0 * (unit[i] * l3) * (unit[i] * link_2_4[i])),
                    6,
                )
            )

    rel_th2 = rel_th2_1 + rel_th2_2  # (rel_th2 = link2와 link3의 사이각도 )
    th2 = th1 + np.pi + rel_th2  # (th2 = ground 기반 절대 각도 )

    # EE_1
    EE_x1 = link2_x + EE1 * np.cos(th2)
    EE_y1 = link2_y + EE1 * np.sin(th2)
    EE_x2 = EE_x1 + EE2 * np.cos(th2 + np.pi / 2)
    EE_y2 = EE_y1 + EE2 * np.sin(th2 + np.pi / 2)

    WS = np.zeros([res, 2])  # initialise WS
    WS[:, 0] = EE_x2
    WS[:, 1] = EE_y2
    # path max distance 구하는 부분 (학습 라벨 데이터로 사용 후보)
    max_distance = 0
    max_p = 0
    max_p_opp = 0
    for d in range(res):
        delta = WS - WS[d, :]  # 전체 점에서 d번째 점 차이값들
        delta_length = np.sqrt(
            delta[:, 0] ** 2 + delta[:, 1] ** 2
        )  # 전체 점에 대해 d점과의 거리들
        max_delta = np.max(delta_length)
        if max_delta >= max_distance:
            max_delta = max_delta  # d 점으로부터 거리들중 max 값과 위치
            max_p = np.argmax(delta_length)
            max_p_opp = d
            max_distance = max_delta
    # assert max_distance > 0, "max_distance is zero!"
    if max_distance < 0:
        max_distance = 0

    # WS 출력 및 기어비 보기위함  : 입력 부분
    Input = np.zeros([res, 2])  # initialise Input
    Input[:, 0] = link2_x
    Input[:, 1] = link2_y

    del_WS = np.zeros([res - 1, 2])  # sizing define
    del_Input = np.zeros([res - 1, 2])  # sizing define
    for j in range(res - 1):
        del_WS[j, :] = WS[j + 1, :] - WS[j, :]
        del_Input[j, :] = Input[j + 1, :] - Input[j, :]

    del_WS = np.concatenate((del_WS, del_WS[0, :].reshape(1, -1)), axis=0)  # sizing
    del_WS_step = np.sqrt(del_WS[:, 0] ** 2 + del_WS[:, 1] ** 2)

    del_Input_step = th1[1] - th1[0]  # [rad]
    dist_ratio = del_Input_step / del_WS_step
    if (max_p == 0) and (max_p_opp == 0):
        min_dist_ratio = 0.0
    else:
        min_dist_ratio = max(
            np.min([dist_ratio[max_p:max_p_opp]]), np.min(dist_ratio)
        )  # only path dependent
    return max_distance, min_dist_ratio
