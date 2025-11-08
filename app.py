import streamlit as st
import json, time, getpass, io
import pandas as pd
from openai import OpenAI
from libs.generateTestCase import generate_testcases
from libs.evaluation import calculate_bert_score

# -------------------------------
# Streamlit App Title
# -------------------------------
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("AI Test Case Generator (GPT-4o-mini)")
st.markdown("Generate structured software test cases from use cases and project descriptions.")

# -------------------------------
# Sidebar: Settings
# -------------------------------
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
   
embedding = st.sidebar.checkbox("Use embeddings for context retrieval", value=False)

# -------------------------------
# Main Input Area
# -------------------------------
st.subheader("Provide Project & Use Case Information")

project_description = st.text_area(
    "Project Description",
    placeholder="Describe the system or project here...",
    height=150
)

usecase_input = st.text_area(
    "Use Case Description (JSON or plain text)",
    placeholder='Example: {"name": "Update Personal Info", "steps": [...]}',
    height=200
)

generate_button = st.button("Generate Test Cases")

# -------------------------------
# Processing
# -------------------------------
if generate_button:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not usecase_input.strip():
        st.error("Please provide a use case description.")
    else:
        try:
            st.info("Generating test cases... please wait.")
            start = time.time()
            
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Generate test cases
            result = generate_testcases(usecase=usecase_input, proj_desc=project_description, client=client, embedding=embedding)

            elapsed = time.time() - start

            # -------------------------------
            # Display results
            # -------------------------------
            st.success(f"Test cases generated successfully in {elapsed:.2f} seconds!")

            if isinstance(result, dict) or isinstance(result, list):
                st.json(result)
            else:
                st.code(result, language="json")
                
            col1, col2 = st.columns(2)
            with col1:
            # Download option
                st.download_button(
                    label="Download Test Cases (JSON)",
                    data=json.dumps(result, indent=4),
                    file_name="generated_testcases.json",
                    mime="application/json"
                )
            
            # Convert result to Excel
            output = io.BytesIO()

            # Handle dict or list gracefully
            if isinstance(result, list):
                # Flatten list of dicts into rows
                df = pd.json_normalize(result)
                #print("result is list: ", df)

            elif isinstance(result, dict):
                if "testCases" in result and isinstance(result["testCases"], list):
                    result = result["testCases"]  #  Extract list of testcases
                    #print("result extract the list of testCases", result)
                else:
                    # If it’s a single testcase dict, wrap it in a list
                    result = [result]

            else:
                # If it's a string (e.g., raw JSON text), try to parse or make single column
                try:
                    df = pd.DataFrame(json.loads(result))
                except:
                    df = pd.DataFrame([{"output": result}])
 
            cleaned = []
            for row in result:
                new_row = {}
                for key, val in row.items():
                    if isinstance(val, (dict, list)):
                        # Compact one-line JSON (no newlines)
                        new_row[key] = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
                    else:
                        new_row[key] = val
                cleaned.append(new_row)
            df= pd.DataFrame(cleaned)
            #print("flatten result",df)

            # Write to Excel in memory
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="TestCases")
                worksheet = writer.sheets["TestCases"]
                for col in worksheet.columns:
                    worksheet.column_dimensions[col[0].column_letter].width = 30  # widen columns
                    
            # Rewind buffer to start
            output.seek(0)
            with col2:
                # Download button
                st.download_button(
                    label="Download Test Cases (Excel)",
                    data=output,
                    file_name="generated_testcases.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"Error: {e}")
