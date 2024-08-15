import sys
import os
# get rid of warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# huggingface transformers    
from transformers import pipeline 



# Load a pre-trained model for question answering 
model = pipeline("question-answering", device=1, model="distilbert-base-cased-distilled-squad") # Example usage 

context = """WHO convened its emergency mpox committee amid concerns that a deadlier strain of the virus, clade Ib, had reached four previously unaffected provinces in Africa. This strain had previously been contained to the Democratic Republic of Congo.

Independent experts on the committee met virtually Wednesday to advise WHO Director-General Tedros Adhanom Ghebreyesus on the severity of the outbreak. After that consultation, he announced Wednesday that he had declared a public health emergency of international concern — the highest level of alarm under international health law.

Also known as PHEIC, this is a status given by WHO to “extraordinary events” that pose a public health risk to other countries through the international spread of disease. These outbreaks may require a coordinated international response, according to the organization."""
#https://www.cnn.com/2024/08/14/health/mpox-who-public-health-emergency/index.html
question = "What is PHEIC?"

print("\n"*3)
print(f"The Question: {question}")
print(f"The Answer: {model(question=question, context=context)['answer']}")
print("\n"*3)