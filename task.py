# 关闭socks代理相关环境变量
import os
os.environ.pop("ALL_PROXY", None)  # 关闭socks代理
os.environ.pop("all_proxy", None)

import argparse
parser = argparse.ArgumentParser(description='处理命名参数')
parser.add_argument('--ai2thor_scene_name', type=str, help='输入你的名称')
args = parser.parse_args()
scene_name = args.ai2thor_scene_name

from memory import MemoryItem, MilvusMemory

memory = MilvusMemory(scene_name, db_ip='127.0.0.1')

from agent import ReMEmbRAgent
import os

#os.environ["OPENAI_API_KEY"] = ""
#os.environ["OPENAI_API_BASE_URL"] = ""

agent = ReMEmbRAgent(llm_type='gpt-4o')

agent.set_memory(memory)




import math
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import cv2



# 创建AI2Thor控制器，禁用Unity图像显示
controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene=scene_name,
    width=640,
    height=480,
    fieldOfView=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    renderSemanticSegmentation=False,
    renderObjectImage=False,
    renderClassImage=False,
    renderNormalsImage=False,
    renderImage=True,  # 启用图像渲染
    platform=CloudRendering
) 


def move_agent_to_target(controller, current_pos, current_theta, target_pos, target_theta=None):
    """
    使用简单的A*寻路算法在x-z平面上移动agent到目标位置，避开障碍物，并用cv2显示每一步的图像。
    注意：这里只做了粗略的网格化寻路，实际环境复杂时建议用更专业的导航模块。
    """
    import heapq
    import numpy as np

    # 1. 构建可行走网格
    # 获取可行走区域（walkable points）
    event = controller.step(action="GetReachablePositions")
    reachable_points = event.metadata.get('actionReturn', [])
    if not reachable_points:
        print("无法获取可行走区域，无法导航。")
        return

    # 网格化参数
    grid_size = 0.25  # AI2Thor默认步长
    def to_grid(pos):
        # 只考虑x和z
        return (round(pos[0]/grid_size), round(pos[2]/grid_size))
    def from_grid(grid):
        return [grid[0]*grid_size, 0, grid[1]*grid_size]

    # 构建walkable set
    walkable_set = set()
    for p in reachable_points:
        walkable_set.add(to_grid([p['x'], 0, p['z']]))

    start_grid = to_grid(current_pos)
    goal_grid = to_grid(target_pos)

    # 2. A*寻路
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def astar(start, goal, walkable):
        heap = []
        heapq.heappush(heap, (0+heuristic(start, goal), 0, start, [start]))
        visited = set()
        while heap:
            _, cost, node, path = heapq.heappop(heap)
            if node == goal:
                return path
            if node in visited:
                continue
            visited.add(node)
            for dx, dz in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = (node[0]+dx, node[1]+dz)
                if neighbor in walkable and neighbor not in visited:
                    heapq.heappush(heap, (cost+1+heuristic(neighbor, goal), cost+1, neighbor, path+[neighbor]))
        return None

    path = astar(start_grid, goal_grid, walkable_set)
    if not path:
        print("找不到到目标的可行走路径。")
        return

    # 3. 沿路径移动
    for idx in range(1, len(path)):
        prev = from_grid(path[idx-1])
        curr = from_grid(path[idx])
        dx = curr[0] - prev[0]
        dz = curr[2] - prev[2]
        # 计算需要面向的角度
        move_angle = math.degrees(math.atan2(dx, dz))
        # 获取当前朝向
        event = controller.step(action="Pass")
        agent_meta = event.metadata.get('agent', {})
        current_yaw = agent_meta.get('rotation', {}).get('y', 0.0)
        # 旋转到move_angle
        angle_diff = (move_angle - current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        num_rotations = int(round(angle_diff / 90))
        for _ in range(abs(num_rotations)):
            if num_rotations > 0:
                event = controller.step(action="RotateRight", degrees=90)
            else:
                event = controller.step(action="RotateLeft", degrees=90)
            if hasattr(event, 'cv2img'):
                cv2.imshow("AI2Thor View", event.cv2img)
                cv2.waitKey(100)
        # 前进一步
        event = controller.step(action="MoveAhead")
        if hasattr(event, 'cv2img'):
            cv2.imshow("AI2Thor View", event.cv2img)
            cv2.waitKey(100)

    # 4. 最后调整朝向
    if target_theta is not None:
        # AI2Thor的rotation是y轴角度，单位为度
        current_event = controller.step(action="Pass")
        agent_meta = current_event.metadata.get('agent', {})
        current_yaw = agent_meta.get('rotation', {}).get('y', 0.0)
        # target_theta可能是弧度，也可能是度
        if abs(target_theta) < 10 or abs(target_theta) > 2*math.pi-10:
            target_yaw = math.degrees(target_theta)
        else:
            target_yaw = target_theta
        angle_diff = (target_yaw - current_yaw + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        num_rotations = int(round(angle_diff / 90))
        for _ in range(abs(num_rotations)):
            if num_rotations > 0:
                #  event = controller.step(action="RotateRight", degrees=90)
                event = controller.step(action="RotateRight", degrees=30)
            else:
                # event = controller.step(action="RotateLeft", degrees=90)
                event = controller.step(action="RotateLeft", degrees=30)
            if hasattr(event, 'cv2img'):
                cv2.imshow("AI2Thor View", event.cv2img)
                cv2.waitKey(100)

while True:
    query = input("我是一个机器人，请问需要什么帮助？输入q退出：")

    if query == 'q':
        cv2.destroyAllWindows()
        break

    print('正在思考...')

    response = agent.query(query)
    print(response.text)
    # print(response)

    # 提取目标位置和朝向theta
    target_position = response.position  # [x, y, z]
    target_theta = response.orientation if hasattr(response, 'orientation') else None
    
    # 获取当前agent位姿
    event = controller.step(action="Pass")
    agent_meta = event.metadata.get('agent', {})
    current_position = [
        agent_meta.get('position', {}).get('x', 0.0),
        agent_meta.get('position', {}).get('y', 0.0),
        agent_meta.get('position', {}).get('z', 0.0)
    ]
    current_theta = agent_meta.get('rotation', {}).get('y', 0.0)  # 角度

    if target_position is not None:
        # 执行移动
        move_agent_to_target(controller, current_position, current_theta, target_position, target_theta)

        print("Agent has moved to the target position and orientation.")

    # 移动结束后显示最终视角并等待用户关闭窗口
    final_event = controller.step(action="Pass")
    if hasattr(final_event, 'cv2img'):
        cv2.imshow("AI2Thor View", final_event.cv2img)
        # print("Press any key in the image window to exit.")
        cv2.waitKey(1)
        #cv2.destroyAllWindows()
