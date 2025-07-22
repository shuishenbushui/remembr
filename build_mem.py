import os
#os.environ["OPENAI_API_KEY"] = ""
#os.environ["OPENAI_API_BASE_URL"] = ""
os.environ.pop("ALL_PROXY", None)  # 关闭socks代理
os.environ.pop("all_proxy", None)

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import time
import math

from memory import MemoryItem, MilvusMemory

import argparse
parser = argparse.ArgumentParser(description='处理命名参数')
parser.add_argument('--ai2thor_scene_name', type=str, help='输入你的名称')
args = parser.parse_args()
scene_name = args.ai2thor_scene_name

# 初始化向量数据库
memory = MilvusMemory(scene_name, db_ip='127.0.0.1')
memory.reset()



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



import os
import openai
import base64
def get_caption_from_gpt4o(img):
    """
    使用gpt-4o对图片进行描述，返回caption字符串
    """

    # 将图像编码为JPEG字节流并转为base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

    client = openai.OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    while True:  # 添加循环，避免单次请求失败
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "you are a robot, please describe the scene you see."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "please describe what you see. exmple： I see a desk in the room."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + img_b64
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            # gpt-4o返回的caption
            content = response.choices[0].message.content if response.choices and response.choices[0].message.content else None
            print('gpt-4o caption: ', content)
            if content is not None:
                return content.strip()
            else:
                return "No caption (gpt-4o returned empty content)"
        except Exception as e:
            print("gpt-4o图片描述失败，使用默认caption。错误信息：", e)
            #return "No caption (gpt-4o failed)"


def main():
    print("AI2Thor 键盘控制程序")
    print("使用 WASD 键移动，QE 键旋转，UI 键上下观察，P 键分析存储记忆")
    print("按 ESC 退出")
    
    while True:
        # 获取当前帧
        event = controller.step(action="Pass")
        
        
        # 处理并显示图像
        display_frame = event.cv2img
        cv2.imshow("AI2Thor View", display_frame)
        # 处理键盘输入
        key = cv2.waitKey(0) & 0xFF  # 阻塞等待键盘输入        


        if key == 27:  # ESC键退出
            break
        elif key == ord('w'):
            event = controller.step(action="MoveAhead")
        elif key == ord('s'):
            event = controller.step(action="MoveBack")
        elif key == ord('a'):
            event = controller.step(action="MoveLeft")
        elif key == ord('d'):
            event = controller.step(action="MoveRight")
        elif key == ord('q'):
            event = controller.step(action="RotateLeft", degrees=90)
        elif key == ord('e'):
            event = controller.step(action="RotateRight", degrees=90)
        elif key == ord('u'):
            event = controller.step(action="LookUp")
        elif key == ord('i'):
            event = controller.step(action="LookDown")
        elif key == ord('p'):
            print('分析图像，存储记忆...')
            # 从event中提取数据存入memory_item
            # 提取caption（简单描述）、时间、位置和朝向
            # caption = "I see " + ", ".join([obj['objectType'] for obj in event.metadata.get('objects', []) if obj.get('visible', False)]) if 'objects' in event.metadata else "No objects"
            caption = get_caption_from_gpt4o(event.cv2img)
            # 获取当前时间戳
            current_time = time.time()
            # 获取agent位置
            agent_meta = event.metadata.get('agent', {})
            position = [
                agent_meta.get('position', {}).get('x', 0.0),
                agent_meta.get('position', {}).get('y', 0.0),
                agent_meta.get('position', {}).get('z', 0.0)
            ]
            # 获取agent朝向（以rotation的y分量为主，单位为弧度）
            theta = agent_meta.get('rotation', {}).get('y', 0.0)
            # 角度转弧度
            theta = math.radians(theta)
            memory_item = MemoryItem(
                caption=caption,
                time=current_time,
                position=position,
                theta=theta
            )
            print('memory_item: ', memory_item)
            memory.insert(memory_item)

    
    # 清理资源
    cv2.destroyAllWindows()
    controller.step(action="Done")

if __name__ == "__main__":
    main()
