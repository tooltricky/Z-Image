"""Z-Image Streamlit Web 界面 - 用户友好的图像生成 GUI。"""

import time
import warnings
from io import BytesIO

import streamlit as st
import torch

warnings.filterwarnings("ignore")

from utils import load_from_local_dir, set_attention_backend
from zimage import generate


def adjust_to_multiple_of_16(value):
    """调整数值为16的倍数（四舍五入到最接近的倍数）"""
    return round(value / 16) * 16


@st.cache_resource
def load_model(model_path, device, dtype, compile_model):
    """加载 Z-Image 模型，使用缓存避免重复加载。"""
    with st.spinner("正在加载 Z-Image 模型... 首次加载可能需要几分钟。"):
        components = load_from_local_dir(
            model_path,
            device=device,
            dtype=dtype,
            compile=compile_model
        )
        set_attention_backend("_native_flash")
        return components


def generate_image(components, prompt, height, width, num_steps, guidance, seed, device):
    """使用 Z-Image 模型生成图像。"""
    generator = torch.Generator(device).manual_seed(seed)

    start_time = time.time()
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
    )
    end_time = time.time()

    return images[0], end_time - start_time


def main():
    # 页面配置
    st.set_page_config(
        page_title="Z-Image 图像生成器",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 主标题
    st.title("⚡ Z-Image 图像生成器")
    st.markdown("**基于单流扩散 Transformer 的高效图像生成基础模型**")
    st.markdown("---")

    # 侧边栏 - 模型配置
    with st.sidebar:
        st.header("⚙️ 模型配置")

        model_path = st.text_input(
            "模型路径",
            value="ckpts/Z-Image-Turbo",
            help="Z-Image 模型目录的路径"
        )

        device = st.selectbox(
            "计算设备",
            options=["cuda", "cpu"],
            index=0,
            help="选择计算设备（推荐使用 CUDA 以获得更快速度）"
        )

        dtype_option = st.selectbox(
            "数据类型",
            options=["bfloat16", "float16", "float32"],
            index=0,
            help="模型精度（推荐 bfloat16）"
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        dtype = dtype_map[dtype_option]

        compile_model = st.checkbox(
            "编译模型",
            value=False,
            help="启用模型编译以加快推理速度（首次运行会较慢）"
        )

        st.markdown("---")
        st.header("🎨 生成参数")

        # 图像尺寸
        col1, col2 = st.columns(2)
        with col1:
            height_input = st.number_input(
                "高度",
                min_value=512,
                max_value=2048,
                value=1024,
                step=16,
                help="输出图像高度（将自动调整为16的倍数）"
            )

        with col2:
            width_input = st.number_input(
                "宽度",
                min_value=512,
                max_value=2048,
                value=1024,
                step=16,
                help="输出图像宽度（将自动调整为16的倍数）"
            )

        # 自动调整为16的倍数
        height = adjust_to_multiple_of_16(height_input)
        width = adjust_to_multiple_of_16(width_input)

        # 如果调整后的值与输入不同，显示提示
        if height != height_input or width != width_input:
            st.info(f"💡 尺寸已自动调整为 {height} × {width}（16的倍数）")

        # 推理参数
        num_inference_steps = st.slider(
            "推理步数",
            min_value=1,
            max_value=50,
            value=8,
            help="去噪步数（Turbo 模型推荐 8）"
        )

        guidance_scale = st.slider(
            "引导系数",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="分类器自由引导系数（Turbo 模型推荐 0.0）"
        )

        seed = st.number_input(
            "随机种子",
            min_value=0,
            max_value=2147483647,
            value=42,
            help="用于可重现性的随机种子"
        )

        st.markdown("---")
        st.info("💡 **提示**：使用 Hopper GPU（H100/H800）时，启用模型编译并使用 Flash Attention 可获得最佳速度。")

    # 主内容区域
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.header("📝 提示词输入")

        # 示例提示词
        example_prompts = {
            "汉服女子": (
                "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
                "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
                "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
                "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
            ),
            "写实人像": "一位年轻女性的写实肖像，自然光线，专业摄影，高细节，8k分辨率",
            "奇幻风景": "一个神奇的奇幻风景，漂浮的岛屿，瀑布，神秘迷雾，充满活力的色彩，数字艺术杰作",
            "自定义": ""
        }

        selected_example = st.selectbox(
            "示例提示词",
            options=list(example_prompts.keys()),
            index=0,
            help="选择一个示例提示词或选择"自定义"来编写您自己的提示词"
        )

        if selected_example == "自定义":
            prompt = st.text_area(
                "输入您的提示词",
                value="",
                height=200,
                placeholder="详细描述您想要生成的图像...",
                help="提供您想要创建的图像的详细描述"
            )
        else:
            prompt = st.text_area(
                "提示词",
                value=example_prompts[selected_example],
                height=200,
                help="编辑提示词或选择"自定义"重新开始"
            )

        # 生成按钮
        generate_button = st.button(
            "🎨 生成图像",
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip()
        )

        if not prompt.strip():
            st.warning("⚠️ 请输入提示词以生成图像。")

    with col_right:
        st.header("🖼️ 生成的图像")

        # 图像显示区域
        image_placeholder = st.empty()
        info_placeholder = st.empty()
        download_placeholder = st.empty()

        # 初始化 session state 用于存储生成的图像
        if 'generated_image' not in st.session_state:
            st.session_state.generated_image = None
            st.session_state.generation_time = None

        # 如果有可用的图像，则显示
        if st.session_state.generated_image is not None:
            image_placeholder.image(
                st.session_state.generated_image,
                caption="生成的图像",
                use_container_width=True
            )
            info_placeholder.success(
                f"✅ 图像在 {st.session_state.generation_time:.2f} 秒内生成完成"
            )

            # 下载按钮
            buf = BytesIO()
            st.session_state.generated_image.save(buf, format="PNG")
            download_placeholder.download_button(
                label="⬇️ 下载图像",
                data=buf.getvalue(),
                file_name=f"zimage_output_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            image_placeholder.info("👈 输入提示词并点击"生成图像"开始")

    # 点击按钮时生成图像
    if generate_button:
        if prompt.strip():
            try:
                # 加载模型
                components = load_model(model_path, device, dtype, compile_model)

                # 生成图像
                with st.spinner("🎨 正在生成图像... 请稍候..."):
                    image, gen_time = generate_image(
                        components=components,
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_steps=num_inference_steps,
                        guidance=guidance_scale,
                        seed=seed,
                        device=device
                    )

                # 存储到 session state
                st.session_state.generated_image = image
                st.session_state.generation_time = gen_time

                # 显示图像
                image_placeholder.image(
                    image,
                    caption="生成的图像",
                    use_container_width=True
                )
                info_placeholder.success(
                    f"✅ 图像生成成功！耗时 {gen_time:.2f} 秒"
                )

                # 下载按钮
                buf = BytesIO()
                image.save(buf, format="PNG")
                download_placeholder.download_button(
                    label="⬇️ 下载图像",
                    data=buf.getvalue(),
                    file_name=f"zimage_output_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ 错误：{str(e)}")
                st.exception(e)

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>由 <strong>Z-Image</strong> 驱动 - 通义·模型AI |
        <a href='https://github.com/Tongyi-MAI/Z-Image' target='_blank'>GitHub</a> |
        <a href='https://arxiv.org/abs/2511.22699' target='_blank'>论文</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
