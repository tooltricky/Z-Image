"""Z-Image Streamlit Web Interface - A user-friendly GUI for image generation."""

import time
import warnings
from io import BytesIO

import streamlit as st
import torch

warnings.filterwarnings("ignore")

from utils import load_from_local_dir, set_attention_backend
from zimage import generate


@st.cache_resource
def load_model(model_path, device, dtype, compile_model):
    """Load Z-Image model with caching to avoid reloading."""
    with st.spinner("Loading Z-Image model... This may take a few minutes on first load."):
        components = load_from_local_dir(
            model_path,
            device=device,
            dtype=dtype,
            compile=compile_model
        )
        set_attention_backend("_native_flash")
        return components


def generate_image(components, prompt, height, width, num_steps, guidance, seed, device):
    """Generate image using Z-Image model."""
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
    # Page configuration
    st.set_page_config(
        page_title="Z-Image Generator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main title
    st.title("⚡ Z-Image Generator")
    st.markdown("**An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer**")
    st.markdown("---")

    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        model_path = st.text_input(
            "Model Path",
            value="ckpts/Z-Image-Turbo",
            help="Path to the Z-Image model directory"
        )

        device = st.selectbox(
            "Device",
            options=["cuda", "cpu"],
            index=0,
            help="Select compute device (CUDA recommended for speed)"
        )

        dtype_option = st.selectbox(
            "Data Type",
            options=["bfloat16", "float16", "float32"],
            index=0,
            help="Model precision (bfloat16 recommended)"
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        dtype = dtype_map[dtype_option]

        compile_model = st.checkbox(
            "Compile Model",
            value=False,
            help="Enable model compilation for faster inference (first run will be slower)"
        )

        st.markdown("---")
        st.header("🎨 Generation Parameters")

        # Image dimensions
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input(
                "Height",
                min_value=512,
                max_value=2048,
                value=1024,
                step=64,
                help="Output image height"
            )

        with col2:
            width = st.number_input(
                "Width",
                min_value=512,
                max_value=2048,
                value=1024,
                step=64,
                help="Output image width"
            )

        # Inference parameters
        num_inference_steps = st.slider(
            "Inference Steps",
            min_value=1,
            max_value=50,
            value=8,
            help="Number of denoising steps (8 recommended for Turbo model)"
        )

        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Classifier-free guidance scale (0.0 recommended for Turbo model)"
        )

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            help="Random seed for reproducibility"
        )

        st.markdown("---")
        st.info("💡 **Tip**: For best speed with Hopper GPUs (H100/H800), enable model compilation and use Flash Attention.")

    # Main content area
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.header("📝 Prompt Input")

        # Example prompts
        example_prompts = {
            "Chinese Woman in Hanfu": (
                "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
                "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
                "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
                "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
            ),
            "Photorealistic Portrait": "A photorealistic portrait of a young woman with natural lighting, professional photography, high detail, 8k resolution",
            "Fantasy Landscape": "A magical fantasy landscape with floating islands, waterfalls, mystical fog, and vibrant colors, digital art masterpiece",
            "Custom": ""
        }

        selected_example = st.selectbox(
            "Example Prompts",
            options=list(example_prompts.keys()),
            index=0,
            help="Select an example prompt or choose 'Custom' to write your own"
        )

        if selected_example == "Custom":
            prompt = st.text_area(
                "Enter your prompt",
                value="",
                height=200,
                placeholder="Describe the image you want to generate in detail...",
                help="Provide a detailed description of the image you want to create"
            )
        else:
            prompt = st.text_area(
                "Prompt",
                value=example_prompts[selected_example],
                height=200,
                help="Edit the prompt or select 'Custom' to start fresh"
            )

        # Generate button
        generate_button = st.button(
            "🎨 Generate Image",
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip()
        )

        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate an image.")

    with col_right:
        st.header("🖼️ Generated Image")

        # Image display area
        image_placeholder = st.empty()
        info_placeholder = st.empty()
        download_placeholder = st.empty()

        # Initialize session state for storing generated image
        if 'generated_image' not in st.session_state:
            st.session_state.generated_image = None
            st.session_state.generation_time = None

        # Display existing image if available
        if st.session_state.generated_image is not None:
            image_placeholder.image(
                st.session_state.generated_image,
                caption="Generated Image",
                use_container_width=True
            )
            info_placeholder.success(
                f"✅ Image generated in {st.session_state.generation_time:.2f} seconds"
            )

            # Download button
            buf = BytesIO()
            st.session_state.generated_image.save(buf, format="PNG")
            download_placeholder.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name=f"zimage_output_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            image_placeholder.info("👈 Enter a prompt and click 'Generate Image' to start")

    # Generate image when button is clicked
    if generate_button:
        if prompt.strip():
            try:
                # Load model
                components = load_model(model_path, device, dtype, compile_model)

                # Generate image
                with st.spinner("🎨 Generating image... Please wait..."):
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

                # Store in session state
                st.session_state.generated_image = image
                st.session_state.generation_time = gen_time

                # Display the image
                image_placeholder.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )
                info_placeholder.success(
                    f"✅ Image generated successfully in {gen_time:.2f} seconds!"
                )

                # Download button
                buf = BytesIO()
                image.save(buf, format="PNG")
                download_placeholder.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"zimage_output_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Powered by <strong>Z-Image</strong> - Tongyi-MAI |
        <a href='https://github.com/Tongyi-MAI/Z-Image' target='_blank'>GitHub</a> |
        <a href='https://arxiv.org/abs/2511.22699' target='_blank'>Paper</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
