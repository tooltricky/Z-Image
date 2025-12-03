# Z-Image Streamlit Web 界面使用说明

## 简介

这是一个用户友好的 Streamlit Web 界面，用于 Z-Image 图像生成模型。通过简单的 Web 界面，您可以轻松地使用 Z-Image-Turbo 模型生成高质量的图像。

## 功能特性

- 🎨 **直观的用户界面**：简洁美观的 Web 界面，易于使用
- 📝 **灵活的提示词输入**：支持自定义提示词和预设示例
- ⚙️ **完整的参数控制**：可调整图像尺寸、推理步数、种子等参数
- 🖼️ **实时图像预览**：生成后立即显示图像
- ⬇️ **一键下载**：支持直接下载生成的图像
- ⚡ **性能优化**：支持模型编译和缓存，加快推理速度
- 🌐 **双语支持**：支持中英文提示词

## 安装依赖

### 1. 安装 Z-Image 依赖

首先安装 Z-Image 的基本依赖：

```bash
pip install -e .
```

### 2. 安装 Streamlit

安装 Streamlit 用于 Web 界面：

```bash
pip install streamlit
```

或者使用完整的依赖列表（如果提供了 requirements-streamlit.txt）：

```bash
pip install -r requirements-streamlit.txt
```

## 准备模型

在运行 Web 界面之前，请确保已下载 Z-Image-Turbo 模型并放置在正确的位置：

1. 从 Hugging Face 或 ModelScope 下载模型：
   - Hugging Face: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
   - ModelScope: https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo

2. 将模型文件放置在 `ckpts/Z-Image-Turbo` 目录下（或其他您指定的路径）

## 启动 Web 界面

在项目根目录下运行以下命令：

```bash
streamlit run app.py
```

默认情况下，应用将在浏览器中自动打开（通常是 `http://localhost:8501`）。

如果需要指定端口或其他配置：

```bash
streamlit run app.py --server.port 8080
```

## 使用指南

### 1. 模型配置（侧边栏）

在左侧边栏中配置模型参数：

- **Model Path**：模型文件路径（默认：`ckpts/Z-Image-Turbo`）
- **Device**：选择计算设备
  - `cuda`：使用 GPU（推荐，速度快）
  - `cpu`：使用 CPU（较慢）
- **Data Type**：模型精度
  - `bfloat16`：推荐，性能与质量平衡
  - `float16`：更节省显存
  - `float32`：最高精度，但消耗更多资源
- **Compile Model**：启用模型编译以加速推理（首次运行会较慢）

### 2. 生成参数（侧边栏）

调整图像生成参数：

- **Height / Width**：输出图像的尺寸（512-2048 像素）
  - 推荐：1024x1024
  - 注意：更大的尺寸需要更多显存和时间
- **Inference Steps**：推理步数（1-50）
  - 推荐：8（Turbo 模型优化的步数）
  - 更多步数可能提高质量，但会增加生成时间
- **Guidance Scale**：分类器自由引导比例（0.0-10.0）
  - 推荐：0.0（Turbo 模型优化设置）
- **Random Seed**：随机种子（0-2147483647）
  - 使用相同的种子和参数可以重现相同的图像

### 3. 输入提示词

在主界面左侧：

1. **选择示例提示词**：
   - 从下拉菜单中选择预设的示例提示词
   - 或选择 "Custom" 自定义您的提示词

2. **编辑或输入提示词**：
   - 在文本框中输入或编辑您的提示词
   - 支持中英文混合
   - 提示：详细的描述通常能生成更好的图像

### 4. 生成图像

1. 确保提示词不为空
2. 点击 **"🎨 Generate Image"** 按钮
3. 等待模型加载和图像生成
   - 首次运行需要加载模型，可能需要几分钟
   - 后续生成会更快（得益于模型缓存）
4. 生成完成后，图像会显示在右侧

### 5. 下载图像

图像生成后，点击 **"⬇️ Download Image"** 按钮即可下载 PNG 格式的图像。

## 提示词编写技巧

### 优质提示词的要素

1. **具体详细**：描述具体的细节，而非笼统的概念
   - ✅ 好：`A young woman with long black hair, wearing a red traditional Chinese dress`
   - ❌ 差：`A woman`

2. **包含风格描述**：指定艺术风格或渲染方式
   - 例如：`photorealistic`, `digital art`, `oil painting`, `anime style`

3. **描述光照和氛围**：
   - 例如：`soft lighting`, `golden hour`, `dramatic shadows`, `neon lights`

4. **添加质量标签**：
   - 例如：`high detail`, `8k resolution`, `professional photography`, `masterpiece`

### 示例提示词

**摄影风格**：
```
A professional portrait of a young Asian woman, natural lighting, shallow depth of field,
high detail, photorealistic, 85mm lens, bokeh background
```

**中国风格**：
```
一位身穿红色汉服的年轻女子，精美的刺绣细节，优雅的发髻，手持团扇，
柔和的光线，背景是中国古典园林，高清细节，专业摄影
```

**幻想风格**：
```
A magical forest with glowing mushrooms, floating lanterns, mystical fog,
vibrant colors, fantasy art style, highly detailed, digital painting masterpiece
```

## 性能优化建议

### 对于高端 GPU（H100/H800/A100）

1. 启用 **Compile Model** 选项
2. 在代码中使用 Flash Attention（需要额外安装）
3. 使用 `bfloat16` 精度

预期性能：
- 首次生成：较慢（需要编译模型）
- 后续生成：可达到亚秒级（< 1 秒）

### 对于消费级 GPU（RTX 3090/4090 等）

1. 使用 `bfloat16` 或 `float16` 精度
2. 推荐图像尺寸：1024x1024
3. 如果显存不足，可以尝试：
   - 降低图像尺寸到 768x768 或 512x512
   - 不启用模型编译

### 对于有限显存设备

如果遇到显存不足（OOM）错误：
1. 降低图像尺寸
2. 使用 `float16` 精度
3. 关闭其他占用 GPU 的程序

## 常见问题

### Q: 启动时提示找不到模型？
A: 确保模型路径正确，并且模型文件已完整下载到指定目录。

### Q: 生成速度很慢？
A:
- 确保使用 CUDA（GPU）而非 CPU
- 首次运行会加载模型，较慢是正常的
- 启用模型编译可以加速（首次编译会更慢，但后续会很快）

### Q: 显存不足（CUDA Out of Memory）？
A:
- 降低图像尺寸
- 使用更低的精度（float16）
- 关闭其他占用 GPU 的程序

### Q: 生成的图像质量不理想？
A:
- 优化提示词，添加更多细节描述
- 调整推理步数（增加步数可能提高质量）
- 尝试不同的随机种子

### Q: 支持批量生成吗？
A: 当前版本暂不支持批量生成，每次生成一张图像。

## 技术细节

- **框架**：Streamlit
- **模型**：Z-Image-Turbo (6B 参数)
- **推理后端**：PyTorch Native
- **推荐显存**：16GB+
- **支持的图像格式**：PNG

## 进阶使用

### 自定义配置

您可以修改 `app.py` 中的默认值来自定义界面：

```python
# 修改默认模型路径
model_path = st.text_input(
    "Model Path",
    value="your/custom/path",  # 修改这里
    ...
)

# 修改默认参数
height = st.number_input(
    "Height",
    value=768,  # 修改默认高度
    ...
)
```

### 集成到现有项目

您可以将生成逻辑提取出来，集成到自己的项目中：

```python
from app import load_model, generate_image

# 加载模型
components = load_model("ckpts/Z-Image-Turbo", "cuda", torch.bfloat16, False)

# 生成图像
image, gen_time = generate_image(
    components=components,
    prompt="Your prompt here",
    height=1024,
    width=1024,
    num_steps=8,
    guidance=0.0,
    seed=42,
    device="cuda"
)

# 保存图像
image.save("output.png")
```

## 相关链接

- Z-Image GitHub: https://github.com/Tongyi-MAI/Z-Image
- Z-Image Paper: https://arxiv.org/abs/2511.22699
- Hugging Face Demo: https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo
- ModelScope Demo: https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo

## 反馈与贡献

如果您在使用过程中遇到问题或有改进建议，欢迎提交 Issue 或 Pull Request。

---

祝您使用愉快！✨
