# FaceDetection-Recognition
This project features a Streamlit application designed to facilitate interaction with documents through natural language processing. Users can upload PDF and DOCX files, which are then processed to extract text, chunked into manageable segments, and stored for future querying.

Key Components:

**1) File Upload and Processing**:

Upload Functionality: Users can upload PDF and DOCX files. The application supports multiple file uploads.
Text Extraction: Text is extracted from PDF and DOCX files using the PyPDF2 and python-docx libraries, respectively.
Text Chunking: The extracted text is split into chunks using langchain's CharacterTextSplitter for efficient querying.
**2) Vectorization and Storage:**

Text Embeddings: The extracted text chunks are converted into vector embeddings using spaCy's pre-trained language model (en_core_web_md).
Vector Store Management: The embeddings are saved to a file and loaded from disk for efficient querying. Users can process files and update the vector store accordingly.
**3) Query Processing:**

Similarity Search: User queries are embedded into vectors and compared against stored embeddings using cosine similarity. The most relevant text chunk is identified and used to generate responses.
Chat Interface: Users interact with the chatbot through a chat interface, which displays their queries and the bot’s responses.
**4) Error Handling and User Feedback:**

Unsupported File Types: Users are notified if an unsupported file type is uploaded.
Loading/Saving Errors: Errors during vector store loading or saving are handled gracefully with appropriate messages.
** Technologies Used:**

Streamlit: For creating the web application interface.
spaCy: For generating text embeddings.
PyPDF2 & python-docx: For extracting text from PDF and DOCX files.
scikit-learn: For computing cosine similarity between text embeddings.
pickle: For saving and loading the vector store.
**Usage:
**
Upload Files: Users upload PDF or DOCX files.
Process Files: The application processes the files, creates text chunks, and stores embeddings.
Query Documents: Users can ask questions about the content of the uploaded documents, and the chatbot provides responses based on the most relevant text chunks.
