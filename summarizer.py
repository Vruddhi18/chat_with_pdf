import fitz  # For PDF text extraction
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

# Function to extract text from a PDF
pdf_path= "C:/Users/acer/Downloads/Diet_tracker_report.pdf"
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to summarize text using T5
def summarize_text(text, model, tokenizer, max_input_length=512, max_output_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_output_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Function to interact with text using Q&A pipeline
def question_answering(context, question, qa_pipeline):
    response = qa_pipeline(question=question, context=context)
    return response["answer"]

# Main code
def main():
    # Load T5 model and tokenizer
    t5_model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    # Load the Q&A pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")

    # Step 1: Extract text from a PDF
    pdf_path = "C:/Users/acer/Downloads/Diet_tracker_report.pdf"  # Replace with your PDF file
    pdf_text = extract_text_from_pdf(pdf_path)

    # Step 2: Summarize the extracted text
    summary = summarize_text(pdf_text, model, tokenizer)
    print("\nSummary:")
    print(summary)

    # Step 3: Q&A interaction
    print("\nInteractive Q&A:")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = question_answering(pdf_text, question, qa_pipeline)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
