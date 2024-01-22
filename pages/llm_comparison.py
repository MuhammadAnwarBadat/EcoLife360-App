import hashlib
import uuid
from itertools import cycle

import diff_viewer
import pandas as pd
import requests
import streamlit as st
from clarifai.client.app import App  # New import to support list_model function
from clarifai.client.auth import V2Stub, create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from clarifai.client.model import Model  # New import to support list_model function
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from typing import Dict
from clarifai.utils.misc import BackoffIterator
import time


# NOTE: Python SDK will be used when the update to take PAT as input is released. 

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


def local_css(file_name):
  with open(file_name) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def reset_session():
  st.session_state['clicked_completions'] = False


def load_pat():
  if 'CLARIFAI_PAT' not in st.secrets:
    st.error("You need to set the CLARIFAI_PAT in the secrets.")
    st.stop()
  return st.secrets.CLARIFAI_PAT


def get_default_models():
  if 'DEFAULT_MODELS' not in st.secrets:
    st.error("You need to set the default models in the secrets.")
    st.stop()
  models = st.secrets.DEFAULT_MODELS.split(",")
  return models


def get_userapp_scopes(stub: V2Stub, userDataObject):
  userDataObj = resources_pb2.UserAppIDSet(
      user_id=userDataObject.user_id, app_id=userDataObject.app_id)
  response = stub.MyScopes(service_pb2.MyScopesRequest(user_app_id=userDataObj))
  return response


def validate_scopes(required_scopes, userapp_scopes):
  if "All" in userapp_scopes or all(scp in userapp_scopes for scp in required_scopes):
    return True
  return False


def init_sessions():
  if 'clicked_completions' not in st.session_state:
    st.session_state['clicked_completions'] = False
  if 'current_model_id' not in st.session_state:
    st.session_state['current_model_id'] = None
  if 'current_workflow_id' not in st.session_state:
    st.session_state['current_workflow_id'] = None

init_sessions()

local_css("./style.css")

DEBUG = False
completions = []

PROMPT_CONCEPT = resources_pb2.Concept(id="prompt", value=1.0)
INPUT_CONCEPT = resources_pb2.Concept(id="input", value=1.0)
COMPLETION_CONCEPT = resources_pb2.Concept(id="completion", value=1.0)


######Using list all functions to get all llm's######
def list_all_models(filter_by: dict = {},) -> Dict[str, Dict[str, str]]:
  """
    Iterator for all the LLM community models.

    Args:
      filter_by: a dictionary of filters to apply to the list of models.

    Returns:
      API_INFO: dictionary of models information.
    """
  llm_community_models = list(App().list_models(filter_by=filter_by, only_in_app=False))
  API_INFO = {}
  for model_name in llm_community_models:
    model_dict = MessageToDict(model_name.model_info)
    try:
      API_INFO[f"{model_dict['id']}: {model_dict['userId']}"] = dict(
          user_id=model_dict["userId"],
          app_id=model_dict["appId"],
          model_id=model_dict["id"],
          version_id=model_dict["modelVersion"]["id"])
    except IndexError:
      pass
  return API_INFO


Examples = [
    {
        "title": "Snoop Doog Summary",
        "template": """Rewrite the following paragraph as a rap by Snoop Dog.
{input}
""",
        "categories": ["Long Form", "Creative"],
    },
]

# This must be within the display() function.
# auth = ClarifaiAuthHelper.from_streamlit(st)
# stub = create_stub(auth)
# userDataObject = auth.get_user_app_id_proto()

# We need to use this post_input and to create and delete models/workflows.
secrets_auth = ClarifaiAuthHelper.from_streamlit(st)
pat = load_pat()
secrets_auth._pat = pat
secrets_stub = create_stub(secrets_auth)  # installer's stub (PAT)

# Check if the user is logged in. If not, use internal PAT.
module_query_params = st.experimental_get_query_params()
if module_query_params.get("pat", "") == "" and module_query_params.get("token", "") == "":
  unauthorized = True
else:
  unauthorized = False
# Get the auth from secrets first and then override that if a pat is provided as a query param.
# If no PAT is in the query param then the resulting auth/stub will match the secrets_auth/stub.
user_or_secrets_auth = ClarifaiAuthHelper.from_streamlit(st)
# This user_or_secrets_stub wil be used for all the predict calls so we bill the user for those.
user_or_secrets_stub = create_stub(user_or_secrets_auth)  # user's (viewer's) stub
userDataObject = user_or_secrets_auth.get_user_app_id_proto()


needed_read_scopes = ['Inputs:Get', 'Models:Get', 'Concepts:Get', 'Predict', 'Workflows:Get']
myscopes_response = get_userapp_scopes(user_or_secrets_stub, userDataObject)
if not validate_scopes(needed_read_scopes, myscopes_response.scopes):
  st.error("You do not have correct scopes for this module")
  st.stop()

# If the user has these write scopes, it means that the user has installed this module in his app. 
# Note: this step is necessary as in secrets.toml, we have PAT for from Clarifai org's account which won't have write scopes for user's installed module.
needed_write_scopes = ['Models:Add', 'Workflows:Add', 'Concepts:Add', 'Models:Delete', 'Workflows:Delete']
if validate_scopes(needed_write_scopes, myscopes_response.scopes):
  secrets_auth = user_or_secrets_auth
  secrets_stub = user_or_secrets_stub


filter_by = dict(
    query="LLM",
    # model_type_id="text-to-text",
)

API_INFO = list_all_models(filter_by=filter_by)

default_llms = get_default_models()

st.markdown(
    "<h1 style='text-align: center; color: black;'>LLM Comparison Toolbox 🧰</h1>",
    unsafe_allow_html=True,
)


def get_user():
  req = service_pb2.GetUserRequest(user_app_id=resources_pb2.UserAppIDSet(user_id="me"))
  response = user_or_secrets_stub.GetUser(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("GetUser request failed: %r" % response)
  return response.user


if unauthorized:
  caller_id = "Anonymous"
else:
  user = get_user()
  caller_id = user.id


# Created models/workflows will be public if the app is public and private if the app is private. 
# If the app is public, it means it is meant for other users to use and hence the models/workflows created by the user should be public as well for them to be accessible.
def get_app_visibility():
  visibility_dict = {'50': resources_pb2.Visibility.Gettable.PUBLIC, '10': resources_pb2.Visibility.Gettable.PRIVATE}
  req = service_pb2.GetAppRequest(user_app_id=userDataObject)
  response = secrets_stub.GetApp(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("GetApp request failed: %r" % response)
  return visibility_dict[str(response.app.visibility.gettable)]

app_visibility = get_app_visibility()


def validate_workflow(cache):
  wkfl_id = st.session_state.get("current_workflow_id")
  if wkfl_id is None:
    return True
  if wkfl_id == -1:
    return False
  resp = secrets_stub.GetWorkflow(service_pb2.GetWorkflowRequest(user_app_id=userDataObject, workflow_id=st.session_state.current_workflow_id))
  if resp.status.code != status_code_pb2.SUCCESS:
    return False
  return True

def validate_model(cache):
  model_id = st.session_state.get("current_model_id")
  if model_id is None:
    return True
  if model_id == -1:
    return False
  resp = secrets_stub.GetModel(service_pb2.GetModelRequest(user_app_id=userDataObject, model_id=st.session_state.current_model_id))
  if resp.status.code != status_code_pb2.SUCCESS:
    return False
  return True


@st.cache_resource(validate=validate_model)
def create_prompt_model(prompt):
  model_id = "test-prompt-model-" + uuid.uuid4().hex[:3]
  response = secrets_stub.PostModels(
      service_pb2.PostModelsRequest(
          user_app_id=userDataObject,
          models=[
              resources_pb2.Model(
                  id=model_id,
                  model_type_id="prompter",
                  visibility=resources_pb2.Visibility(gettable=app_visibility)
              ),
          ],
      ))

  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModels request failed: %r" % response)
  
  req = service_pb2.PostModelVersionsRequest(
      user_app_id=userDataObject,
      model_id=model_id,
      model_versions=[resources_pb2.ModelVersion(
            output_info=resources_pb2.OutputInfo(),
            visibility=resources_pb2.Visibility(gettable=app_visibility)
          )
      ],
  )
  params = json_format.ParseDict(
      {
          "prompt_template": prompt,
          # "position": position,
      },
      req.model_versions[0].output_info.params,
  )
  post_model_versions_response = secrets_stub.PostModelVersions(req)
  if post_model_versions_response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelVersions request failed: %r" % post_model_versions_response)

  return post_model_versions_response.model


def delete_model(model_id):
  response = secrets_stub.DeleteModels(
      service_pb2.DeleteModelsRequest(
          user_app_id=userDataObject,
          ids=[model_id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteModels request failed: %r" % response)
  st.success(f"Deleted model {model_id}")


def create_workflows(prompt, models):
  workflows = []
  try:
    prompt_model = create_prompt_model(prompt)
    for model in models:
      workflows.append(create_workflow(prompt_model, model))
  except Exception as e:
    st.session_state.current_model_id = -1
    prompt_model = create_prompt_model(prompt)
    st.session_state.current_model_id = None
    for model in models:
      workflows.append(create_workflow(prompt_model, model))

  st.success(
      f"Created {len(workflows)} workflows! Now ready to test it out!")
  return prompt_model, workflows
  
    

@st.cache_resource(validate=validate_workflow)
def create_workflow(prompt_model, selected_llm):
  metadata = {'llm': selected_llm}
  metadata_struct = Struct()
  metadata_struct.update(metadata)
  req = service_pb2.PostWorkflowsRequest(
      user_app_id=userDataObject,
      workflows=[
          resources_pb2.Workflow(
              id=
              f"test-workflow-{API_INFO[selected_llm]['user_id']}-{API_INFO[selected_llm]['model_id']}-"
              + uuid.uuid4().hex[:3],
              metadata = metadata_struct,
              nodes=[
                  resources_pb2.WorkflowNode(
                      id="prompt",
                      model=resources_pb2.Model(
                          id=prompt_model.id,
                          user_id=prompt_model.user_id,
                          app_id=prompt_model.app_id,
                          model_version=resources_pb2.ModelVersion(
                              id=prompt_model.model_version.id,
                              user_id=prompt_model.user_id,
                              app_id=prompt_model.app_id,
                          ),
                      ),
                  ),
                  resources_pb2.WorkflowNode(
                      id="llm",
                      model=resources_pb2.Model(
                          id=API_INFO[selected_llm]["model_id"],
                          user_id=API_INFO[selected_llm]["user_id"],
                          app_id=API_INFO[selected_llm]["app_id"],
                          model_version=resources_pb2.ModelVersion(
                              id=API_INFO[selected_llm]["version_id"],
                              user_id=API_INFO[selected_llm]["user_id"],
                              app_id=API_INFO[selected_llm]["app_id"],
                          ),
                      ),
                      node_inputs=[resources_pb2.NodeInput(node_id="prompt",)],
                  ),
              ],
              visibility=resources_pb2.Visibility(gettable=app_visibility),
          ),
      ],
  )

  response = secrets_stub.PostWorkflows(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostWorkflows request failed: %r" % response)
  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response.workflows[0]


def delete_workflow(workflow_id: str):
  response = secrets_stub.DeleteWorkflows(
      service_pb2.DeleteWorkflowsRequest(
          user_app_id=userDataObject,
          ids=[workflow.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteWorkflows request failed: %r" % response)
  else:
    st.success(f"Deleted workflow {workflow_id}")


@st.cache_resource
@st.cache_data
def run_workflow(input_text, workflow):
  start_time = time.time()
  backoff_iterator = BackoffIterator()
  while True:
    response = user_or_secrets_stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=userDataObject,
            workflow_id=workflow.id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
            ],
        ))
    
    if response.status.code == status_code_pb2.MODEL_DEPLOYING and \
      time.time() - start_time < 60 * 10: # 10 minutes
      st.info(f"Model is still deploying, please wait...")
      time.sleep(next(backoff_iterator))
      continue

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(f"PostWorkflowResults failed with response {response.status!r}")
    else:
      break

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response.results[0].outputs[1].data.text.raw


def get_prompt_model(pmodel):
  resp = secrets_stub.GetModel(service_pb2.GetModelRequest(user_app_id=userDataObject, model_id=pmodel.id))
  if resp.status.code != status_code_pb2.SUCCESS:
    return False
  return True

def get_workflow(wkfl):
  resp = secrets_stub.GetWorkflow(service_pb2.GetWorkflowRequest(user_app_id=userDataObject, workflow_id=wkfl.id))
  if resp.status.code != status_code_pb2.SUCCESS:
    return False
  return True

def check_model_workflows(prompt_model, workflows, prompt, models):
  if not get_prompt_model(prompt_model):
    st.session_state.current_model_id = -1
    st.session_state.current_workflow_id = -1
    prompt_model, new_wkfls = create_workflows(prompt, models)
  else:
    new_wkfls = []
    for w in workflows:
      if not get_workflow(w):
        st.session_state.current_workflow_id = -1
        workflows.append(create_workflow(prompt_model, w.metadata.fields['llm'].string_value))
        st.session_state.current_workflow_id = None
      else:
        new_wkfls.append(w)
  st.session_state.current_model_id = None
  st.session_state.current_workflow_id = None
  return prompt_model, new_wkfls


@st.cache_resource
def post_input(txt, id, concepts=[], metadata=None):
  """Posts input to the API and returns the response."""
  metadata_struct = Struct()
  metadata_struct.update(metadata)
  metadata = metadata_struct
  try:
    input_job_id = Inputs(
        logger_level="ERROR", user_id=userDataObject.user_id,
        app_id=userDataObject.app_id, pat=secrets_auth._pat).upload_from_bytes(id, text_bytes=bytes(txt, 'UTF-8'), labels=concepts, metadata=metadata)

  except Exception as e:
    st.error(f"post input error:{e}")
    #st.stop

  return input_job_id


def list_concepts():
  """Lists all concepts in the user's app."""
  response = secrets_stub.ListConcepts(service_pb2.ListConceptsRequest(user_app_id=userDataObject,))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("ListConcepts request failed: %r" % response)
  return response.concepts


def post_concept(concept):
  """Posts a concept to the user's app."""
  response = secrets_stub.PostConcepts(
      service_pb2.PostConceptsRequest(
          user_app_id=userDataObject,
          concepts=[concept],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostConcepts request failed: %r" % response)
  return response.concepts[0]


def search_inputs(concepts=[], metadata=None, page=1, per_page=20):
  """Searches for inputs in the user's app."""
  req = service_pb2.PostAnnotationsSearchesRequest(
      user_app_id=userDataObject,
      searches=[resources_pb2.Search(query=resources_pb2.Query(filters=[]))],
      pagination=service_pb2.Pagination(
          page=page,
          per_page=per_page,
      ),
  )
  if len(concepts) > 0:
    req.searches[0].query.filters.append(
        resources_pb2.Filter(
            annotation=resources_pb2.Annotation(data=resources_pb2.Data(concepts=concepts,))))
  if metadata is not None:
    req.searches[0].query.filters.append(
        resources_pb2.Filter(
            annotation=resources_pb2.Annotation(data=resources_pb2.Data(metadata=metadata,))))
  response = secrets_stub.PostAnnotationsSearches(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("SearchInputs request failed: %r" % response)
  return response


# def get_text(url):
#   """Download the raw text from the url"""
#   response = requests.get(url)
#   return response.text


def get_text(auth, url):
  """Download the raw text from the url"""
  try:
    h = {"Authorization": f"Key {auth._pat}"}
    response = requests.get(url, headers=h)
    response.encoding = response.apparent_encoding
  except Exception as e:
    print(f"Error: {e}")
    response = None
  return response.text if response else ""


# Check if prompt, completion and input are concepts in the user's app
app_concepts = list_concepts()
for concept in [PROMPT_CONCEPT, INPUT_CONCEPT, COMPLETION_CONCEPT]:
  if concept.id not in [c.id for c in app_concepts]:
    st.warning(
        f"The {concept.id} concept is not in your app. Please add it by clicking the button below."
    )
    if st.button(f"Add {concept.id} concept"):
      post_concept(concept)
      st.experimental_rerun()

app_concepts = list_concepts()
app_concept_ids = [c.id for c in app_concepts]

# Check if all required concepts are in the app
concepts_ready_bool = True
for concept in [PROMPT_CONCEPT, INPUT_CONCEPT, COMPLETION_CONCEPT]:
  if concept.id not in app_concept_ids:
    concepts_ready_bool = False

# Check if all required concepts are in the app
if not concepts_ready_bool:
  st.error("Need to add all the required concepts to the app before continuing.")
  st.stop()

prompt_search_response = search_inputs(concepts=[PROMPT_CONCEPT], per_page=12)
completion_search_response = search_inputs(concepts=[COMPLETION_CONCEPT], per_page=12)
user_input_search_response = search_inputs(concepts=[INPUT_CONCEPT], per_page=12)

st.markdown(
    "<h2 style='text-align: center; color: #667085;'>Recent prompts from others</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align: center;'>Hover to copy and try them out yourself!</div>",
    unsafe_allow_html=True,
)


def create_next_completion_gen(completion_search_response, user_input_search_response, input_id):
  for completion_hit in completion_search_response.hits:
    if completion_hit.input.data.metadata.fields["input_id"].string_value == input_id:
      for user_input_hit in user_input_search_response.hits:
        if (completion_hit.input.data.metadata.fields["user_input_id"].string_value ==
            user_input_hit.input.id):
          yield completion_hit.input, user_input_hit.input


previous_prompts = []

# Check if gen dict is in session state
if "completion_gen_dict" not in st.session_state:
  st.session_state.completion_gen_dict = {}
  st.session_state.first_run = True

completion_gen_dict = st.session_state.completion_gen_dict

cols = cycle(st.columns(3))
for idx, prompt_hit in enumerate(prompt_search_response.hits):
  txt = get_text(secrets_auth, prompt_hit.input.data.text.url)
  previous_prompts.append({
      "prompt": txt,
  })
  container = next(cols).container()
  metadata = json_format.MessageToDict(prompt_hit.input.data.metadata)
  caller_id = metadata.get("caller", "zeiler")
  if caller_id == "":
    caller_id = "zeiler"

  if len(completion_gen_dict) < len(prompt_search_response.hits):
    completion_gen_dict[prompt_hit.input.id] = create_next_completion_gen(
        completion_search_response, user_input_search_response, prompt_hit.input.id)

  container.subheader(f"Prompt ({caller_id})", anchor=False)
  container.code(txt)  # metric(label="Prompt", value=txt)

  container.subheader("Answer", anchor=False)

  # Create persistent placeholders to update when user clicks next
  st.session_state[f"placeholder_model_name_{prompt_hit.input.id}"] = container.empty()
  st.session_state[f"placeholder_user_input_{prompt_hit.input.id}"] = container.empty()
  st.session_state[f"placeholder_completion{prompt_hit.input.id}"] = container.empty()

  if container.button("Next", key=prompt_hit.input.id):
    try:
      completion_input, user_input = next(completion_gen_dict[prompt_hit.input.id])

      completion_text = get_text(secrets_auth, completion_input.data.text.url)
      user_input_text = get_text(secrets_auth, user_input.data.text.url)
      model_url = completion_input.data.metadata.fields["model"].string_value

      st.session_state[f"placeholder_model_name_{prompt_hit.input.id}"].markdown(
          f"Generated by {model_url}")
      st.session_state[f"placeholder_user_input_{prompt_hit.input.id}"].markdown(
          f"**Input**:\n {user_input_text}")
      st.session_state[f"placeholder_completion{prompt_hit.input.id}"].markdown(
          f"**Completion**:\n {completion_text}")
    except StopIteration:
      completion_gen_dict[prompt_hit.input.id] = create_next_completion_gen(
          completion_search_response, user_input_search_response, prompt_hit.input.id)
      st.warning("No more completions available. Starting from the beginning.")

query_params = st.experimental_get_query_params()
prompt = ""
if "prompt" in query_params:
  prompt = query_params["prompt"][0]

st.subheader("Test out new prompt templates with various LLM models")

model_names = sorted(API_INFO.keys())
models = st.multiselect(
    "Select the LLMs you want to use:", model_names, default=default_llms, on_change=reset_session)

prompt = st.text_area(
    "Enter your prompt template to test out here:",
    placeholder="Explain {data.text.raw} to a 5 yeard old.",
    value=prompt,
    on_change=reset_session,
    help=
    "You need to place a placeholder {data.text.raw} in your prompt template. If that is in the middle then two prefix and suffix prompt models will be added to the workflow.",
)

if prompt and models:
  if prompt.find("{data.text.raw}") == -1:
    st.error("You need to place a placeholder {data.text.raw} in your prompt template.")
    st.stop()

  if len(models) == 0:
    st.error("You need to select at least one model.")
    st.stop()

  prompt_model, workflows = create_workflows(prompt, models)

input = st.text_area(
    "Try out your new workflow by providing some input:",
    on_change=reset_session,
    help=
    "This will be used as the input to the {data.text.raw} placeholder in your prompt template.",
)

completion_button = st.button("Generate completions")
if completion_button:
  st.session_state.clicked_completions = True

if st.session_state.get('clicked_completions') and prompt and models and input:
  concepts = list_concepts()
  concept_ids = [c.id for c in concepts]
  for concept in [PROMPT_CONCEPT, INPUT_CONCEPT, COMPLETION_CONCEPT]:
    if concept.id not in concept_ids:
      post_concept(concept)
      st.success(f"Added {concept.id} concept")

  # Add the input as an inputs in the app.
  id = hashlib.md5(input.encode("utf-8")).hexdigest()
  inp_job_id = post_input(
      input, id, concepts=["input"], metadata={
          "tags": ["input"],
          "caller": caller_id
      })
  st.markdown(
      "<h1 style='text-align: center;font-size: 40px;color: #667085;'>Completions</h1>",
      unsafe_allow_html=True,
  )

  prompt_model, workflows = check_model_workflows(prompt_model, workflows, prompt, models)

  for workflow in workflows:
    container = st.container()
    container.write(prompt.replace("{data.text.raw}", input))
    predicted_text = run_workflow(input, workflow)

    model_url = f"https://clarifai.com/{workflow.nodes[1].model.user_id}/{workflow.nodes[1].model.app_id}/models/{workflow.nodes[1].model.id}"
    model_url_with_version = f"{model_url}/versions/{workflow.nodes[1].model.model_version.id}"
    container.write(f"Completion from {model_url}:")

    completion = predicted_text
    container.code(completion)
    completion_job_id = post_input(
        completion,
        id=hashlib.md5(completion.encode("utf-8")).hexdigest(),
        concepts=['completion'],
        metadata={
            "input_id": id,
            "tags": ["completion"],
            "model": model_url_with_version,
            "caller": caller_id,
        },
    )
    completions.append({
        "select":
            False,
        "model":
            model_url,
        "completion":
            completion.strip(),
        "input_id":
            f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/inputs/{completion_job_id}",
    })

  c = pd.DataFrame(completions)
  edited_df = st.data_editor(c, disabled=set(completions[0].keys()) - set(["select"]))

  st.subheader("Diff completions")
  st.markdown("Select two completions to diff")
  selected_rows = edited_df.loc[edited_df['select']]
  if len(selected_rows) != 2:
    st.warning("Please select two completions to diff")
  else:
    old_value = selected_rows.iloc[0]["completion"]
    new_value = selected_rows.iloc[1]["completion"]
    cols = st.columns(2)
    cols[0].markdown(f"Completion: {selected_rows.iloc[0]['model']}")
    cols[1].markdown(f"Completion: {selected_rows.iloc[1]['model']}")
    diff_viewer.diff_viewer(old_text=old_value, new_text=new_value, lang='none')

st.divider()

st.text("When you're finished experimenting, click the below button to help keep the app clean.")
cleanup = st.button("Cleanup workflows and prompt models")
if cleanup:
  # Cleanup so we don't have tons of junk in this app
  for workflow in workflows:
    delete_workflow(workflow.id)
  delete_model(prompt_model.id)
