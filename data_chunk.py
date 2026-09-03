from camel.models import OllamaModel
from camel.agents import ChatAgent
from camel.messages import BaseMessage


def run_chunk(essay):

    # ---------------------------------------
    # 1. Load local Ollama model
    # ---------------------------------------
    model = OllamaModel(
        model_type="qwen2.5:3b",
        model_config_dict={
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        url="http://127.0.0.1:11434/v1",
    )

    # ---------------------------------------
    # 2. Create ChatAgent only once
    # ---------------------------------------
    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=(
                "You split medical text into accurate, "
                "short factual statements. "
                "Do not invent or add information."
            ),
        ),
        model=model,
    )

    # ---------------------------------------
    # 3. Split document into paragraphs
    # ---------------------------------------
    paragraphs = essay.split("\n\n")

    propositions = []

    # ---------------------------------------
    # 4. Process every paragraph
    # ---------------------------------------
    for i, para in enumerate(paragraphs):

        para = para.strip()

        if not para:
            continue

        prompt = f"""
Break the following medical text into short,
self-contained factual statements.

Rules:
- Keep all important medical information.
- Do not invent information.
- Do not add information that is not present.
- Return exactly one factual statement per line.
- Do not number the statements.
- Preserve patient findings.
- Preserve symptoms.
- Preserve diagnoses.
- Preserve treatments and medications.
- Preserve medical tests.
- Preserve test results.
- Preserve dates, measurements, and other important values.
- Keep the original meaning.

Medical text:

{para}
"""

        # ---------------------------------------
        # 5. Create user message
        # ---------------------------------------
        message = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
        )

        # ---------------------------------------
        # 6. Ask Qwen to create propositions
        # ---------------------------------------
        response = agent.step(message)

        text = response.msg.content

        # ---------------------------------------
        # 7. Clean the response
        # ---------------------------------------
        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove common bullet/number formatting
            line = line.lstrip("-•* ")

            # Remove numbering such as:
            # 1. statement
            # 2) statement
            if len(line) > 2 and line[0].isdigit():

                if line[1] in ".):":
                    line = line[2:].strip()

                elif len(line) > 3 and line[1:3] in [". ", ") "]:
                    line = line[3:].strip()

            if line:
                propositions.append(line)

        print(f"Done with paragraph {i}")

    # ---------------------------------------
    # 8. Return all propositions
    # ---------------------------------------
    return propositions


# -------------------------------------------
# TEST
# -------------------------------------------
if __name__ == "__main__":

    test_text = """
The patient has fever and cough. The patient was
diagnosed with pneumonia.

Chest X-ray showed an infiltrate.
"""

    result = run_chunk(test_text)

    print("\n========== PROPOSITIONS ==========\n")

    for i, proposition in enumerate(result, start=1):
        print(f"{i}. {proposition}")