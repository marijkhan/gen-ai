import streamlit as st
import requests
import json
import base64

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="LinkedIn Post Generator", page_icon="📝", layout="wide")
st.title("LinkedIn Content Curation Agent")
st.markdown("Generate engaging LinkedIn posts using an AI-powered agentic system.")

topic = st.text_input("Enter a topic for your LinkedIn post:", placeholder="e.g., GenAI agents for backend engineers")

col1, col2 = st.columns(2)
plan_only = col1.button("Generate Plan Only")
generate = col2.button("Generate Post")

if plan_only and topic:
    with st.spinner("Planning..."):
        try:
            resp = requests.post(f"{API_BASE}/plan", json={"topic": topic}, timeout=60)
            resp.raise_for_status()
            plan = resp.json()

            st.subheader("Execution Plan")
            for step in plan["steps"]:
                deps = f" (depends on: {step['depends_on']})" if step["depends_on"] else " (no dependencies)"
                st.markdown(f"**Step {step['step']}** — `{step['tool']}`{deps}\n\n{step['description']}")

            with st.expander("Raw Plan JSON"):
                st.json(plan)
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

if generate and topic:
    with st.spinner("Generating plan and executing... this may take a minute."):
        try:
            resp = requests.post(f"{API_BASE}/execute", json={"topic": topic}, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            # Main content area
            st.subheader("Generated LinkedIn Post")
            if data.get("final_post"):
                st.text_area("Post Content", value=data["final_post"], height=300)
            else:
                st.warning("No post was generated. Check the debug panel below for details.")

            if data.get("image_base64"):
                st.subheader("Generated Image")
                try:
                    image_bytes = base64.b64decode(data["image_base64"])
                    st.image(image_bytes, use_container_width=True)
                except Exception:
                    st.warning("Failed to decode image.")
            else:
                st.info("No image was generated (image generation may have failed or was not in the plan).")

            # Debug panel
            with st.expander("Debug: Execution Plan"):
                st.json(data["plan"])

            with st.expander("Debug: Execution Waves"):
                for i, wave in enumerate(data.get("execution_order", [])):
                    steps_in_wave = []
                    for step_num in wave:
                        for s in data["plan"]["steps"]:
                            if s["step"] == step_num:
                                steps_in_wave.append(f"Step {step_num}: {s['tool']}")
                    st.markdown(f"**Wave {i + 1}**: {', '.join(steps_in_wave)}")

            with st.expander("Debug: Step Results"):
                for result in data.get("results", []):
                    status_icon = "✅" if result["status"] == "success" else "❌"
                    st.markdown(f"{status_icon} **Step {result['step']}** — `{result['tool']}` — {result['duration_ms']}ms")
                    if result.get("error"):
                        st.error(result["error"])
                    elif result["tool"] != "image_generator":
                        preview = result["output"][:500] + "..." if len(result["output"]) > 500 else result["output"]
                        st.text(preview)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

if not topic and (plan_only or generate):
    st.warning("Please enter a topic first.")
