from dotenv import load_dotenv
import json
import logging
import logging.config
import os
import re
import pandas as pd
import streamlit as st
import uuid
import yaml
from services import bedrock_agent_runtime

load_dotenv()

# Configure logging using YAML
# if os.path.exists("logging.yaml"):
#     with open("logging.yaml", "r") as file:
#         config = yaml.safe_load(file)
#         logging.config.dictConfig(config)
# else:
    #log_level = logging.getLevelNamesMapping()[(os.environ.get("LOG_LEVEL", "INFO"))]
    #logging.basicConfig(level=log_level)

#logger = logging.getLogger(__name__)

agent_id = os.environ.get("BEDROCK_AGENT_ID")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "Welcome to AutoMDR Agent..")
ui_icon = os.environ.get("BEDROCK_AGENT_TEST_UI_ICON")
region_name='us-east-1'


def init_session_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}
    st.session_state.pop("uploaded_file", None)


st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)
if len(st.session_state.items()) == 0:
    init_session_state()

# Sidebar: File Upload + Reset
with st.sidebar:
    st.subheader("Upload a file (optional)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "json", "xls", "xlsx"])

    if uploaded_file:
        file_type = uploaded_file.type
        file_bytes = uploaded_file.read()
        file_content = ""

        try:
            if uploaded_file.name.endswith(".txt"):
                file_content = file_bytes.decode("utf-8")
                st.text_area("Text Preview", file_content, height=200)

            elif uploaded_file.name.endswith(".json"):
                json_data = json.loads(file_bytes.decode("utf-8"))
                file_content = json.dumps(json_data, indent=2)
                st.json(json_data)

            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                excel_data = pd.read_excel(uploaded_file, sheet_name=None)
                sheet_names = list(excel_data.keys())

                st.markdown("### Available Sheets:")
                selected_sheets = st.multiselect(
                    "Select sheets to include in the prompt",
                    sheet_names,
                    default=sheet_names
                )

                combined_data = ""
                for sheet in selected_sheets:
                    df = excel_data[sheet]
                    st.markdown(f"#### Preview: {sheet}")
                    st.dataframe(df)
                    combined_data += f"\n\n--- Sheet: {sheet} ---\n"
                    combined_data += df.to_csv(index=False)

                file_content = combined_data

            elif uploaded_file.name.endswith(".pdf"):
                file_content = "[PDF uploaded - content not previewed]"
                st.success("PDF uploaded successfully.")

            elif uploaded_file.name.endswith(".docx"):
                file_content = "[DOCX uploaded - content not previewed]"
                st.success("DOCX uploaded successfully.")

            else:
                file_content = "[Unsupported file type]"
                st.warning("Preview not available.")

            st.session_state.uploaded_file = {
                "name": uploaded_file.name,
                "content": file_content,
                "type": uploaded_file.type
            }

        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Reset Session"):
        init_session_state()

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat + Upload Handling
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    file_prompt_suffix = ""
    if "uploaded_file" in st.session_state:
        file_name = st.session_state.uploaded_file["name"]
        file_text = st.session_state.uploaded_file["content"]
        file_prompt_suffix = f"\n\n[Attached File: {file_name}]\n{file_text}"

    full_prompt = prompt + file_prompt_suffix

    with st.chat_message("assistant"):
        with st.empty():
            with st.spinner():
                response = bedrock_agent_runtime.invoke_agent(
                    agent_id,
                    agent_alias_id,
                    region_name,
                    st.session_state.session_id,
                    full_prompt
                )
            output_text = response["output_text"]

            try:
                output_json = json.loads(output_text, strict=False)
                if "instruction" in output_json and "result" in output_json:
                    output_text = output_json["result"]
            except json.JSONDecodeError:
                pass

            if len(response["citations"]) > 0:
                citation_num = 1
                output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
                citation_locs = ""
                for citation in response["citations"]:
                    for retrieved_ref in citation["retrievedReferences"]:
                        citation_marker = f"[{citation_num}]"
                        citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['s3Location']['uri']}"
                        citation_num += 1
                output_text += f"\n{citation_locs}"

            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.session_state.citations = response["citations"]
            st.session_state.trace = response["trace"]
            st.markdown(output_text, unsafe_allow_html=True)

# Trace Viewer
trace_types_map = {
    "Pre-Processing": ["preGuardrailTrace", "preProcessingTrace"],
    "Orchestration": ["orchestrationTrace"],
    "Post-Processing": ["postProcessingTrace", "postGuardrailTrace"]
}

trace_info_types_map = {
    "preProcessingTrace": ["modelInvocationInput", "modelInvocationOutput"],
    "orchestrationTrace": ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"],
    "postProcessingTrace": ["modelInvocationInput", "modelInvocationOutput", "observation"]
}

with st.sidebar:
    st.title("Trace")
    step_num = 1
    for trace_type_header in trace_types_map:
        st.subheader(trace_type_header)
        has_trace = False
        for trace_type in trace_types_map[trace_type_header]:
            if trace_type in st.session_state.trace:
                has_trace = True
                trace_steps = {}
                for trace in st.session_state.trace[trace_type]:
                    if trace_type in trace_info_types_map:
                        trace_info_types = trace_info_types_map[trace_type]
                        for trace_info_type in trace_info_types:
                            if trace_info_type in trace:
                                trace_id = trace[trace_info_type]["traceId"]
                                trace_steps.setdefault(trace_id, []).append(trace)
                                break
                    else:
                        trace_id = trace["traceId"]
                        trace_steps[trace_id] = [{trace_type: trace}]
                for trace_id in trace_steps:
                    with st.expander(f"Trace Step {step_num}", expanded=False):
                        for trace in trace_steps[trace_id]:
                            trace_str = json.dumps(trace, indent=2)
                            st.code(trace_str, language="json", line_numbers=True, wrap_lines=True)
                    step_num += 1
        if not has_trace:
            st.text("None")

    st.subheader("Citations")
    if len(st.session_state.citations) > 0:
        citation_num = 1
        for citation in st.session_state.citations:
            for retrieved_ref_num, retrieved_ref in enumerate(citation["retrievedReferences"]):
                with st.expander(f"Citation [{citation_num}]", expanded=False):
                    citation_str = json.dumps({
                        "generatedResponsePart": citation["generatedResponsePart"],
                        "retrievedReference": retrieved_ref
                    }, indent=2)
                    st.code(citation_str, language="json", line_numbers=True, wrap_lines=True)
                citation_num += 1
    else:
        st.text("None")
