from camel.models import OllamaModel
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO


model = OllamaModel(
    model_type="qwen2.5:3b",
    model_config_dict={
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    url="http://127.0.0.1:11434/v1",
)

agent = KnowledgeGraphAgent(model=model)

uio = UnstructuredIO()

text = """
The patient is a 65-year-old male with fever and cough.
He was diagnosed with pneumonia.
The patient was treated with antibiotics.
Chest X-ray showed an infiltrate in the right lung.
"""

element = uio.create_element_from_text(text=text)

print("Sending text to Qwen...")
print(text)

result = agent.run(
    element,
    parse_graph_elements=True
)

print("\n========== RESULT ==========")
print(result)

print("\n========== NODES ==========")
print(result.nodes)

print("\n========== RELATIONSHIPS ==========")
print(result.relationships)