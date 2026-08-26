from camel.models import OllamaModel
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from typing import List


def run_chunk(essay):

    model = OllamaModel(
        model_type="qwen2.5:3b",
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        url="http://127.0.0.1:11434/v1",
    )

    paragraphs = essay.split("\n\n")

    propositions = []

    for i, para in enumerate(paragraphs):

        if not para.strip():
            continue

        prompt = f"""
Break the following medical text into short, self-contained factual statements.

Rules:
- Keep all important medical information.
- Do not invent information.
- Return one statement per line.
- Do not number the statements.
- Preserve patient findings, symptoms, diagnoses, treatments, tests and results.

Medical text:

{para}
"""

        message = BaseMessage.make_user_message(
            role_name="User",
            content=prompt
        )

        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Assistant",
                content="You split medical text into accurate factual statements."
            ),
            model=model
        )

        response = agent.step(message)

        text = response.msg.content

        for line in text.splitlines():
            line = line.strip()

            if line:
                line = line.lstrip("-•0123456789. ")
                if line:
                    propositions.append(line)

        print(f"Done with {i}")

    return propositions