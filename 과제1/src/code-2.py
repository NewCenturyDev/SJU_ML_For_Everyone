import pandas as pd
import numpy as np
import sympy as syp

# CSV 파일을 읽어 들인다
data = pd.read_csv("data.csv", encoding='utf-8')

# X1 열을 x1List 배열에 담는다
x1List = data["X1 kernel_area"].astype("float")
# X2 열을 x2List 배열에 담는다 (형변환시 string으로 표현된 숫자의 ","가 변환 오류를 유발하는 이슈로 for문 추가)
x2List = []
for (idx, x) in enumerate(data["X2 kernel_length"]):
    x2List.append(float(data["X2 kernel_length"].get(idx).replace(",", "")))
# 분류 열을 y배열에 담는다
yList = data["Wheat Varieties"]


# 4주차 4-5강의의 Multiclass Classification의 One-vs-All 방식을 사용하여 3가지 분류에 따른 로지스틱 회귀를 수행 한다
# Sigmoid 함수와 그 도함수를 나타낸다. 범위는 -1~1이므로 a=1로 한다
def g(z, a):
    return 1/(1 + np.exp(-a * z))


def diff_g(z):
    return g(z, 1) * (1 - g(z, 1))


# 4주차 4-1강의의 Sigmoid 함수를 사용해서 Wheat Varieties가 Kama인 것과 아닌 것으로 분류한다
def h(x1, x2, w0, w1, w2):
    return g(w0 + w1 * x1 + w2 * x2, 1)


def cost(hxi, yi):
    return -yi * np.log(hxi) - (1-yi) * np.log(1 - hxi)


def j(m):
    sum_of_cost = 0
    for i in range(m):
        # 초기 w값은 w0 = 0, w1 = 1, w2 = 1로 한다
        sum_of_cost += cost(h(x1List[i], x2List[i], 0, 1, 1), yList[i] == "Kama")
    return 1 / m * sum_of_cost


def diff_j_for_w1(m):
    sum_of_cost = 0
    for i in range(m):
        # 초기 w값은 w0 = 0, w1 = 1, w2 = 1로 한다
        sum_of_cost += cost(h(x1List[i], x2List[i], 0, 1, 1), yList[i] == "Kama")
    return 1 / m * sum_of_cost

"""
# 3주차 3-2강의의 특징값 스케일링 - 정규화 방법 (Normalization Feature)에 따라 정규화(평균 정규화)를 수행 한다
# 정규화된 x1 특징값 데이터들이 담길 배열이다
x1NormalizedList = []
# 정규화된 x2 특징값 데이터들이 담길 배열이다
x2NormalizedList = []

# x1의 평균값과 다이나믹 레인지(최댓값 - 최솟값) 을 구한다
x1Max = np.max(x1List)
x1Min = np.min(x1List)
x1Avg = np.average(x1List)
x1DynamicRange = x1Max - x1Min

# x2의 평균값과 다이나믹 레인지(최댓값 - 최솟값) 을 구한다
x2Max = np.max(x2List)
x2Min = np.min(x2List)
x2Avg = np.average(x2List)
x2DynamicRange = x2Max - x2Min

# x1 특징값들을 정규화한다
for x1 in x1List:
    x1NormalizedList.append((x1 - x1Avg) / (x1Max - x1Min))

# x2 특징값들을 정규화한다
for x2 in x2List:
    x2NormalizedList.append((x2 - x2Avg) / (x2Max - x2Min))

# 데이터프레임으로 변환한다
resultDf = pd.DataFrame({
    "Normalized X1 kernel_area": x1NormalizedList,
    "Normalized X2 kernel_length": x2NormalizedList
})

# 결과값을 콘솔에 출력한다 (확인용)
print("Normalized X1 kernel_area")
print(x1NormalizedList)
print("Normalized X2 kernel_length")
print(x2NormalizedList)
print(resultDf)

# CSV에 작성한다
resultDf.to_csv("normalized_data.csv", sep=",", na_rep="NaN", index=False)
"""