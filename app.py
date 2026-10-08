# app.py
import os
import torch
import logging
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------ 日志配置 ------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NAIP-Backend")

# ------------ 配置区域 ------------
WEIGHT_DIR = "/Users/ruyi/Desktop/NAIP/last"
MAX_LENGTH = 512
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ FastAPI 初始化 ------------
app = FastAPI(title="NAIP Paper Score Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ 加载模型 & Tokenizer ------------
try:
    logger.info("Loading tokenizer and model from %s...", WEIGHT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHT_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        WEIGHT_DIR,
        local_files_only=True,
        num_labels=1,
        # load_in_8bit=True  # 保持 8bit 加载
    )
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    logger.info("Model loaded successfully on %s.", MODEL_DEVICE)
except Exception as e:
    logger.error("Failed to load model: %s", str(e))
    raise RuntimeError("Model initialization failed.")

# ------------ 边界限制常量 (确保这些已定义) ------------
TITLE_MIN_CHAR = 10
TITLE_MAX_CHAR = 300
ABSTRACT_MIN_CHAR = 100
ABSTRACT_MAX_CHAR = 6000
MIN_WORDS_TITLE = 3
MIN_WORDS_ABSTRACT = 20

# ------------ 请求体模型与多重边界检测 ------------
from pydantic import BaseModel, field_validator
import re 

class PredictRequest(BaseModel):
    title: str
    abstract: str

    @field_validator('title', 'abstract')
    @classmethod
    def comprehensive_validate(cls, v: str, info):
        field_name = info.field_name.capitalize() # 获取当前校验的是 title 还是 abstract
        
        # --- D. 格式自动化清洗 (Sanitization) ---
        # 1. 将换行符替换为空格 (处理 PDF 复制带来的硬换行)
        v = v.replace('\n', ' ').replace('\r', ' ')
        # 2. 去除前后空格
        v = v.strip()
        # 3. 将多个连续空格合并为一个 (可选，但建议加上)
        v = re.sub(r'\s+', ' ', v)

        # --- C. 内容质量/多样性校验 (Content Diversity) ---
        # 检查前 100 个字符（如果不足 100 则检查全部）
        sample = v[:100]
        if len(sample) > 20: # 只有长度达到一定程度才检测多样性，防止误伤极短的合法标题
            unique_chars = len(set(sample))
            if unique_chars < 5:
                raise ValueError(f"{field_name} contains too many repetitive characters (low diversity)")

        # --- A. 字符长度检测 ---
        char_count = len(v)
        if field_name == "Title":
            if char_count < 10: raise ValueError("Title must be at least 10 characters")
            if char_count > 300: raise ValueError("Title is too long (max 300)")
        else: # Abstract
            if char_count < 100: raise ValueError("Abstract must be at least 100 characters")
            if char_count > 6000: raise ValueError("Abstract is too long (max 6000)")

        # --- B. 单词数量检测 ---
        word_count = len(v.split())
        if field_name == "Title":
            if word_count < 3: raise ValueError("Title must contain at least 3 words")
        else: # Abstract
            if word_count < 20: raise ValueError("Abstract must contain at least 20 words")

        # 返回清洗后的字符串，后续推理将使用这个 v
        return v

# ------------ 核心：自定义异常处理器 (解决 Value error 前缀问题) ------------
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    该函数会拦截所有 Pydantic 校验错误，并格式化输出。
    它会遍历所有错误，去掉 "Value error, " 前缀。
    """
    error_messages = []
    for error in exc.errors():
        msg = error.get("msg", "")
        # 清洗掉 Pydantic V2 自动添加的前缀
        clean_msg = msg.replace("Value error, ", "").replace("Assertion failed, ", "")
        error_messages.append(clean_msg)
    
    # 将多个错误合并成一个字符串返回，或者返回列表
    return JSONResponse(
        status_code=422,
        content={"detail": " ; ".join(error_messages)}, 
    )

# ------------ 核心逻辑 ------------
PROMPT_TEMPLATE = (
    "Given a certain paper, Title: {title}\n"
    "Abstract: {abstract}.\n"
    "Predict its normalized academic impact (between 0 and 1):"
)

def predict_score(title: str, abstract: str) -> float:
    # 构造 prompt
    input_text = PROMPT_TEMPLATE.format(
        title=title.replace("\n", " "),
        abstract=abstract.replace("\n", " ")
    )
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    # 将 tensor 移动到设备
    inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 假设是回归任务，使用 Sigmoid 映射到 0-1
        score = torch.sigmoid(outputs.logits).item()
    
    return float(score)

# ------------ 路由 ------------
@app.post("/predict")
async def predict(req: PredictRequest):
    start_time = time.time()
    logger.info("Received prediction request for title: %s...", req.title[:50])
    
    try:
        score = predict_score(req.title, req.abstract)
        duration = time.time() - start_time
        
        logger.info("Inference successful. Score: %.4f, Duration: %.2fs", score, duration)
        
        return {
            "status": "success",
            "score": round(score, 6),
            "meta": {
                "title_len": len(req.title),
                "abstract_len": len(req.abstract),
                "duration": round(duration, 2)
            }
        }
    except Exception as e:
        logger.error("Inference error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal model inference error.")

# ------------ 静态资源处理 ------------
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def read_index():
        index_file = frontend_dir / "index.html"
        return index_file.read_text(encoding="utf-8")
else:
    @app.get("/")
    async def hello():
        return {"msg": "NAIP Paper Score Inference Service is running. Frontend directory not found."}

if __name__ == "__main__":
    import uvicorn
    # 使用 workers=1 因为 8bit 模型加载通常不适合多进程并行在同一张显卡
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=False)