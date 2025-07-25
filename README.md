# 🤖 Unofficial Simplified Implementation of NVIDIA's ReMEmbR in AI2-THOR Environment 🧠
ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robots 
## 🎥 [Bilibili 视频：机器人演示](https://www.bilibili.com/video/BV1nRbkzmEZd/?spm_id_from=333.1387.homepage.video_card.click&vd_source=f5aceb5f4e7793d3e5cabca8dcfa32ed)
## 🛠️ setup
1.Install Python dependencies
```
conda activate remembr
python -m pip install -r requirements.txt
pip install ai2thor
```
2.Install MilvusDB
```
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
```

## 🚀 run
0.Switching to Conda environment
```
source activate remembr
```
1.Start vector database MilvusDB
```
bash launch_milvus_container.sh start
```
2.build memory
```
export OPENAI_API_KEY="your OPENAI_API_KEY"
export OPENAI_API_BASE_URL="your OPENAI_API_BASE_URL"
python build_mem.py --ai2thor_scene_name=FloorPlan10
```
Control the robot to roam around and collect and analyze images \
3.query
```
export OPENAI_API_KEY = "your OPENAI_API_KEY"
export OPENAI_API_BASE_URL = "your OPENAI_API_BASE_URL"
python task.py --ai2thor_scene_name=FloorPlan10
```
Then you can ask the agent any questions 


    

