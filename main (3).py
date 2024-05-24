import json
import re
from typing import List
from fastapi import FastAPI, UploadFile, File
import networkx as nx
import ast
import matplotlib.pyplot as plt
import numpy as np
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import os
from joblib import load
from fastapi import HTTPException
#from fpdf import FPDF
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app = FastAPI()

desired_order = ['post', 'get', 'put', 'delete']

def custom_sort(item):
    http_method = list(list(item.values())[0].keys())[0]
    return desired_order.index(http_method)


def check_for_simple_get_endpoint(string, key):
    pattern = r'\{.*\}'
    if not bool(re.search(pattern, string)):
        if isinstance(key, dict) and list(key.keys())[0] == "get":
            return True
    return 


def check_for_simple_get_post_endpoints(string, key):
    pattern = r'\{.*\}'
    if re.search(pattern, string) and "get" in key:
        return True
    elif "post" in key:
        return True
    return False


def pnc(apiflow):
    for seq in apiflow['sequences']:
        simple_get_endp = []
        sequence_counter = 2
        for endp in seq['sequence_1']:
            for k, v in endp.items():
                if check_for_simple_get_endpoint(k, v):
                    simple_get_endp.append(endp)
        if len(simple_get_endp) >= 1:
            SG = simple_get_endp[0]
            for endp in seq['sequence_1']:
                for k, v in endp.items():
                    if check_for_simple_get_endpoint(k, v) or check_for_simple_get_post_endpoints(k, v):
                        continue
                    EP = endp
                    sequence_key = f"sequence_{sequence_counter}"
                    seq[sequence_key] = [SG, EP]
                    sequence_counter += 1
    return apiflow


os.environ["OPENAI_API_KEY"] = "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"


loader = PyPDFLoader("resources2.pdf")
docs = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
template = """Use the following pieces of context to answer the question at the end.
Use only python list format, no explanation no text and only re arranged resources.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


def re_arranging_resources(resource):
    prom="from the given list of resources of an API re arrange them as per their working sequence, additionally strict guidelines are the size of resultant list of resources should be same as given list"
    ans = rag_chain.invoke(str(resource)+prom)
    my_list = ast.literal_eval(ans)
    if len(resource)!=len(my_list):
        resp = llm.invoke(str(resource)+prom)
        my_list = ast.literal_eval(resp.content)
    return my_list


def generate_flowchart(list_of_resources, save_path):
    G = nx.DiGraph()
    for i in range(len(list_of_resources) - 1):
        G.add_edge(list_of_resources[i], list_of_resources[i+1])
    plt.figure(figsize=(10, 6))
    num_colors = len(list_of_resources)
    colors = [plt.cm.viridis(i) for i in np.linspace(0.3, 1, num_colors)] #modify the color darkness using this line 0-0.3
    color_map = {resource: colors[i] for i, resource in enumerate(list_of_resources)}
    pos = nx.spring_layout(G, pos={node: (i, -i) for i, node in enumerate(list_of_resources)})
    nx.draw(G, pos, with_labels=True, node_color=[color_map[node] for node in list_of_resources], node_size=1000, font_size=12, font_weight='bold')
    plt.title("API Resource Flowchart")
    plt.savefig(save_path)
    plt.close()


# def generate_descriptive_pdf(list_of_resources, rearranged_list, api_flow):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="The resources of the API spec are as follows:", ln=True, align="C")
#     pdf.ln(10)
#     for i, resource in enumerate(list_of_resources, start=1):
#         pdf.cell(200, 10, txt=f"{i}. {resource}", ln=True)
#     pdf.ln(10)
#     pdf.cell(200, 10, txt="The resources of the API spec now as per AI-bot are as follows:", ln=True, align="C")
#     pdf.ln(10)
#     for i, resource in enumerate(rearranged_list, start=1):
#         pdf.cell(200, 10, txt=f"{i}. {resource}", ln=True)
#     pdf.add_page()
#     pdf.ln(10)
#     pdf.cell(200, 10, txt="Possible API resource flows from AI-bot:", ln=True, align="C")
#     pdf.ln(10)
#     pdf.image("api_flow_chart.png", x=10, y=pdf.y, w=180)
#     pdf.add_page()
#     pdf.ln(10)
#     pdf.cell(200, 10, txt="The sequential API flow for the resources as per the AI-bot is as follows:", ln=True, align="C")
#     pdf.ln(10)
#     pdf.multi_cell(0, 10, txt=json.dumps(api_flow, indent=4), align="L")
#     pdf.output("api_descriptive.pdf")


def text_preprocess(endpoint: str) -> List[str]:
    patterns = ["api/v1", "/api/v1","api/v2/", "/api/v2/","/api/v3", "api/v3","/api/v4","/api/v4", "/v1", "/v2", "/v3", "/v4", "/api","/"]
    for pattern in patterns:
        endpoint = endpoint.replace(pattern, " ")
    cleaned_endpoint = re.sub(r'[^a-zA-Z-_]', ' ', endpoint)
    cleaned_endpoint = [word.replace('_', ' ').replace('-', ' ') for word in cleaned_endpoint.split()]
    return cleaned_endpoint


SEnew_model = load('SEnew_model')


def predict_endp(endp):
    item1 = text_preprocess(endp)
    item1 = ' '.join(item1)
    se = SEnew_model.predict([item1.lower()])
    se = True if se[0] == 1 else False
    return se


def predict_params(params: List[str]) -> List[str]:
    se_list = []
    for item in params:
        item1 = text_preprocess(item)
        item1 = ' '.join(item1)
        se = SEnew_model.predict([item1.lower()])
        if se[0] == 1:
            se_list.append(item)
    return se_list


def categorize_sensitivity(se_count: int) -> str:
    if se_count >= 3:
        return "HIGH"
    elif se_count <= 1:
        return "LOW"
    elif 1 < se_count <= 4:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_sensitivity(api_flow):
    for resource in api_flow['sequences']:
        params = []
        for endpoint_data in resource['sequence_1']:
            endpoint = list(endpoint_data.keys())[0]
            params.append(endpoint)
        se_params = predict_params(params)
        sensitivity = categorize_sensitivity(len(se_params))
        resource['Sensitivity'] = sensitivity

        seqs = [key for key in resource.keys() if 'sequence' in key]
        for seq in seqs:
            for endp in resource[seq]:
                endp1 = list(endp.keys())[0]
                se =predict_endp(endp1)
                endp['sensitive'] = se
    return api_flow

        
            
def spec_parsing(spec_file):
    try:
        data = spec_file
        if 'paths' not in data or not data['paths']:
            raise ValueError("Invalid API specification: 'paths' key is missing or empty")
        
        api_flow = {
            "name": "API Spec file",
            "description": "Represents the common flow of operations in an API Spec file based on analysis of the OpenAPI Specification.",
            "sequences": []
        }
        
        list_of_resources = []
        paths = data['paths']
        
        for endpoint, methods in paths.items():
            for http_method, details in methods.items():
                if http_method in ['get', 'post', 'put', 'delete']:
                    if "tags" not in details:
                        raise ValueError(f"Missing 'tags' key in endpoint details: {endpoint}")
                    if not details["tags"]:
                        raise ValueError(f"Empty 'tags' list in endpoint details: {endpoint}")
                    resource_tag = details["tags"][0]
                    if resource_tag not in list_of_resources:
                        list_of_resources.append(resource_tag)
        
        if len(list_of_resources) > 1:
            rearranged_list = re_arranging_resources(list_of_resources)
        else:
            rearranged_list = list_of_resources
        
        for resource in rearranged_list:
            ress = {
                'resource_name': resource,
                'Sensitivity': "",
                'sequence_1': []
            }
            for endpoint, methods in paths.items():
                for http_method, details in methods.items():
                    if http_method in ['get', 'post', 'put', 'delete'] and details.get("tags") and details["tags"][0] == resource:
                        ress['sequence_1'].append({endpoint: {http_method: paths[endpoint][http_method]}})
            ress['sequence_1'] = sorted(ress['sequence_1'], key=custom_sort)
            api_flow["sequences"].append(ress)
        
        api_flow = pnc(api_flow)
        api_flow = calculate_sensitivity(api_flow)
        generate_flowchart(rearranged_list, "api_flow_chart.png")
        #generate_descriptive_pdf(list_of_resources, rearranged_list, api_flow)
        
        with open("api_flow1.json", "w") as json_output_file:
            json_output_file.write(json.dumps(api_flow, indent=4))
        
        return api_flow, "api_descriptive.pdf"
    
    except KeyError as e:
        if str(e) == "'paths'":
            raise ValueError("Invalid API specification: 'paths' key is missing") from e
        elif str(e) == "'tags'":
            raise ValueError(f"Missing 'tags' key in endpoint details: {endpoint}") from e
        else:
            raise ValueError("Invalid API specification format") from e





@app.post("/generate-API-flow")
async def generate_result(json_file: UploadFile = File(...)):
    try:
        data = json.load(json_file.file)
        api_flow_json, descriptive_pdf = spec_parsing(data)
        return api_flow_json
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))