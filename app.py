import os
import io
import time
import base64
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入我们昨天跑通的核心类
from run_official_demo import MarigoldV2

app = FastAPI(title="Luce AI Engine API", version="1.0.0")

# 允许跨域（方便手机端和网页端无缝调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动适配模型路径（优先读环境变量，其次匹配阿里云挂载路径，最后退回 AutoDL 本地路径）
def get_model_paths():
    base_candidates = [
        os.getenv("BASE_MODEL_URI"),
        "/workspace/models/Qwen-Image-Edit-2509",
        "/root/autodl-tmp/marigold-v2/assets/checkpoints/Qwen-Image-Edit-2509",
    ]
    model_candidates = [
        os.getenv("MODEL_URI"),
        "/workspace/models/Marigold-V2",
        "/root/autodl-tmp/marigold-v2/assets/checkpoints/Marigold-V2",
    ]
    
    base_uri = next((p for p in base_candidates if p and os.path.exists(p)), None)
    model_uri = next((p for p in model_candidates if p and os.path.exists(p)), None)
    
    return base_uri, model_uri

model = None

@app.on_event("startup")
def startup_event():
    """容器启动时执行一次，把大模型常驻在 4090 / A10 显存中"""
    global model
    base_uri, model_uri = get_model_paths()
    print(f"🌟 [Luce Engine] 初始化硬件设备: {DEVICE}")
    print(f"📦 [Luce Engine] 探测 Base 模型路径: {base_uri}")
    print(f"📦 [Luce Engine] 探测 Marigold 路径: {model_uri}")
    
    if not base_uri or not model_uri:
        print("⚠️ [警告] 未检测到本地模型文件，请确保挂载路径正确！")
        return

    print("🚀 正在加载 Marigold V2 核心模型进显存...")
    start_time = time.time()
    model = MarigoldV2(base_uri, model_uri, DEVICE)
    print(f"✅ 模型常驻显存成功！耗时: {time.time() - start_time:.2f} 秒")

# 辅助函数：把 PIL 图像转为 Base64 字符串给手机
def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 1. 健康检查接口（云平台用于检测服务是否正常启动）
@app.get("/")
@app.get("/healthz")
def health_check():
    return {
        "status": "ready" if model is not None else "loading_or_missing_weights",
        "device": str(DEVICE),
        "service": "Luce AI Surface Analyzer"
    }

# 2. 核心分析接口：接收手机手绘图，返回 Normals / Albedo / Depth
@app.post("/v1/ai/analyze-surface")
async def analyze_surface(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="模型正在初始化或权重路径未找到，请检查挂载！")
    
    try:
        start_time = time.time()
        
        # 读取手机上传的图像字节流
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 跑核心推理（单步流匹配，2秒完成）
        results = model(raw_image)
        
        duration = time.time() - start_time
        print(f"⚡ [推理完成] 尺寸: {raw_image.size}, 耗时: {duration:.2f}s")
        
        # 组装返回给手机 App 的数据包
        return {
            "status": "success",
            "duration_seconds": round(duration, 2),
            "normals": pil_to_base64(results["Normals"]),
            "albedo": pil_to_base64(results["Albedo"]),
            "depth": pil_to_base64(results["Depth"])
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 本地或容器内直接运行测试
    uvicorn.run(app, host="0.0.0.0", port=8000)