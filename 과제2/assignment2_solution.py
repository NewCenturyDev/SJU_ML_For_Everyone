import os
import platform
import numpy
import matplotlib.pyplot as plt

data_points = []
cluster = []


centroid1 = {
    "x":-1,
    "y":1
}
centroid2 = {
    "x":1,
    "y":-0.5
}

def get_cluster_assignment():
    return [datapoint["cluster"] for datapoint in data_points]

def get_diff(data_point=None, show_result=True):
    diff = {
        "x": None,
        "y": None
    }
    
    if data_point is None:
        diff["x"] = eval(input("point_x: "))
        diff["y"] = eval(input("point_y: "))
    else:
        diff = data_point

    distance_to_c1 = ((centroid1["x"]-diff["x"])**2+(centroid1["y"]-diff["y"])**2)**(1/2)
    distance_to_c2 = ((centroid2["x"]-diff["x"])**2+(centroid2["y"]-diff["y"])**2)**(1/2)
    member_of_cluster = -1
    if distance_to_c1 < distance_to_c2:
        member_of_cluster = 1
    else:
        member_of_cluster = 2
    print("******Result*******")
    print("Distance to C1: " + str(distance_to_c1))
    print("Distance to C2: " + str(distance_to_c2))
    print("This point is member of cluster " + str(member_of_cluster))
    print("*******************")
    return member_of_cluster

def get_mean(c_no):
    new_centroid = {
        "x": None,
        "y": None,
    }
    sum_x = 0
    sum_y = 0
    cnt = 0
    
    for data in data_points:
        if "cluster" in data:
            if data["cluster"] is c_no:
                sum_x += data["x"]
                sum_y += data["y"]
                cnt += 1
    new_centroid["x"] = sum_x / cnt
    new_centroid["y"] = sum_y / cnt
    return new_centroid

def run_kmean_cycle():
    cluster_assignment = get_cluster_assignment()
    print("클러스터 할당 정보: " + str(cluster_assignment))
    global centroid1
    centroid1 = get_mean(1)
    global centroid2
    centroid2 = get_mean(2)
    print("새로운 센트로이드 1: " + str(centroid1))
    print("새로운 센트로이드 2: " + str(centroid2))
    for point in data_points:
            point["cluster"] = get_diff(point, False)
    return cluster_assignment

def add_data_point():
    point = {
        "x": None,
        "y": None,
        "cluster": None
    }
    point["x"] = eval(input("point_x: "))
    point["y"] = eval(input("point_y: "))
    point["cluster"] = get_diff(point, False)
    data_points.append(point)

def draw_graph():
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    idx = 0
    for point in data_points:
        idx += 1
        plt.scatter(point["x"], point["y"], label=idx)
    plt.scatter(centroid1["x"], centroid1["y"], label="centroid1")
    plt.scatter(centroid2["x"], centroid2["y"], label="centroid2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='x-large')
    plt.show()
    

while(True):
    command = None
    print("******모두를위한머신러닝 과제2 해주는 프로그램*******")
    print("명령-데이터셋에 점 추가: add")
    print("     (오입력시 프로그램을 재실행 하고 처음부터 다시 입력)")
    print("명령-센트로이드까지의 거리 계산 및 할당된 클러스터 보기: calc")
    print("명령-센트로이드 이동 1회 수행: move")
    print("명령-현재 데이터/센트로이드  좌표평면 표시: graph")
    print("명령-KMEAN 알고리즘 실행 및 최종결과 좌표평면: kmean")
    print("명령-프로그램 종료: exit")
    print("******모두를위한머신러닝 과제2 해주는 프로그램*******")

    command = input("input command: ")
    #IDEL 쉘은 클리어가 안되네...
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

    if command == "add":
        add_data_point()
    elif command == "calc":
        get_diff()
    elif command == "move":
        for point in data_points:
            point["cluster"] = get_diff(point, False)
        current_assigned_cluster = run_kmean_cycle()
    elif command == "graph":
        draw_graph()
    elif command == "kmean":
        for point in data_points:
            point["cluster"] = get_diff(point, False)
        current_assigned_cluster = run_kmean_cycle()
        while(current_assigned_cluster!=get_cluster_assignment()):
            current_assigned_cluster = run_kmean_cycle()
    elif command == "exit":
        os._exit(os.EX_OK)
    else:
        print("잘못된 명령")
