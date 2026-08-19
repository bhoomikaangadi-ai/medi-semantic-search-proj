from concurrent.futures import ThreadPoolExecutor

from camel.models import OllamaModel
from camel.agents import ChatAgent
from camel.messages import BaseMessage


# ============================================================
# Ollama Model
# ============================================================

model = OllamaModel(
    model_type="qwen2.5:3b",
    model_config_dict={
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    url="http://127.0.0.1:11434/v1",
)


# ============================================================
# Summary Prompt
# ============================================================

sum_prompt = """
Generate a structured summary from the provided medical text.

IMPORTANT RULES:
- Use ONLY information explicitly present in the text.
- Do NOT invent or assume information.
- Do NOT add information from outside the text.
- Keep the output concise and factual.
- Return ONLY categories that contain relevant information.

Use the following categories when relevant:

ANATOMICAL_STRUCTURE:
Anatomical structures specifically discussed.

BODY_FUNCTION:
Body functions discussed.

BODY_MEASUREMENT:
Measurements such as blood pressure, temperature,
oxygen saturation, ejection fraction, etc.

BM_RESULT:
Result of the measurement.

BM_UNIT:
Unit of measurement.

BM_VALUE:
Numerical value.

LABORATORY_DATA:
Laboratory tests mentioned.

LAB_RESULT:
Results of laboratory tests.

LAB_VALUE:
Numerical laboratory values.

LAB_UNIT:
Units of laboratory measurements.

MEDICINE:
Medications mentioned.

MED_DOSE:
Medication dose.

MED_DURATION:
Medication duration.

MED_FORM:
Medication form.

MED_FREQUENCY:
Medication frequency.

MED_ROUTE:
Medication route.

MED_STATUS:
Whether the medication was started, stopped,
continued, or discontinued.

MED_STRENGTH:
Medication strength.

MED_UNIT:
Medication unit.

MED_TOTALDOSE:
Total medication dose if explicitly mentioned.

PROBLEM:
Medical conditions, diagnoses, symptoms, or findings.

PROCEDURE:
Medical procedures performed.

PROCEDURE_RESULT:
Results of procedures.

PROC_METHOD:
Methods used.

SEVERITY:
Severity of medical conditions.

MEDICAL_DEVICE:
Medical devices used.

SUBSTANCE_ABUSE:
Substance abuse information, only if explicitly mentioned.

FORMAT:

CATEGORY_NAME: information

Example:

PROBLEM: Congestive heart failure; cardiorespiratory arrest.

MEDICINE: Metoprolol; captopril.

MED_STATUS: Metoprolol and captopril were prescribed.

BODY_MEASUREMENT: Oxygen saturation.

BM_VALUE: Greater than 95%.

BM_UNIT: Percent; oxygen delivered at 2 liters.

Do not provide explanations outside the categories.
"""


# ============================================================
# Summarize One Chunk
# ============================================================

def call_ollama_api(chunk):

    try:

        agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="Medical Summarizer",
                content=sum_prompt
            ),
            model=model
        )

        message = BaseMessage.make_user_message(
            role_name="User",
            content=chunk
        )

        response = agent.step(message)

        if response is None:
            return ""

        if response.msg is None:
            return ""

        return response.msg.content

    except Exception as e:

        print("ERROR while summarizing chunk:")
        print(e)

        return ""


# ============================================================
# Split Text Into Chunks
# ============================================================

def split_into_chunks(text, words_per_chunk=800):

    words = text.split()

    chunks = []

    for i in range(0, len(words), words_per_chunk):

        chunk = " ".join(
            words[i:i + words_per_chunk]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# Process All Chunks
# ============================================================

def process_chunks(content):

    chunks = split_into_chunks(
        content,
        words_per_chunk=800
    )

    print("\n========================================")
    print("Summary generation")
    print("Number of chunks:", len(chunks))
    print("========================================\n")

    responses = []

    # Keep this low because Qwen is running locally.
    with ThreadPoolExecutor(max_workers=2) as executor:

        futures = []

        for chunk in chunks:

            futures.append(
                executor.submit(
                    call_ollama_api,
                    chunk
                )
            )

        for i, future in enumerate(futures):

            try:

                response = future.result()

                print(
                    f"Summary completed: "
                    f"{i + 1}/{len(chunks)}"
                )

                if response:
                    responses.append(response)

            except Exception as e:

                print(
                    f"ERROR processing summary "
                    f"chunk {i + 1}: {e}"
                )

    return responses


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    content = """
    The patient was admitted with cardiorespiratory arrest.
    The patient had congestive heart failure with an ejection
    fraction of 15% to 20%.

    The patient was treated with metoprolol and captopril.

    Oxygen saturation was greater than 95% on
    2 liters nasal cannula.

    The patient developed acute renal failure which
    resolved following fluid resuscitation.
    """

    results = process_chunks(content)

    print("\n========== SUMMARY ==========\n")

    for i, result in enumerate(results):

        print(f"--- Summary {i + 1} ---")
        print(result)
        print()