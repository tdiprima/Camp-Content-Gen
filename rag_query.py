"""
This script queries a vector store to retrieve information about the UltraLight Tent from the hiking products file.
Author: Tammy DiPrima
"""
from openai import OpenAI
import time
import os
import dotenv

dotenv.load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Your existing vector store ID
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# Retrieve the vector store
vector_store = client.vector_stores.retrieve(VECTOR_STORE_ID)
print(f"Retrieved vector store: {vector_store.name}")

# Create an assistant linked to this vector store
assistant = client.beta.assistants.create(
    name="RAG Assistant",
    instructions="Answer questions based on the documents in the vector store.",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    model="gpt-4o"
)

# Create a thread and ask a question
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What are the features of the UltraLight Tent in the hiking products file?"
)

# Run the assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for it to finish
while True:
    run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run_status.status == "completed":
        break
    print("Waiting for response...")
    time.sleep(2)

# Print out the assistant's answer
messages = client.beta.threads.messages.list(thread_id=thread.id)
print("\nAssistant's response:")
for msg in messages.data:
    if msg.role == "assistant":
        print(msg.content[0].text.value)

# Clean up if you don't need them anymore
client.beta.assistants.delete(assistant.id)
client.beta.threads.delete(thread.id)
print("Cleanup complete: Assistant and Thread deleted.")
