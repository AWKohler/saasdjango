import os

import openai as openai
from PyPDF2 import PdfReader
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rest_framework.decorators import permission_classes, authentication_classes
from gradio_client import Client
from rest_framework_api_key.permissions import HasAPIKey
from rest_framework.decorators import parser_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets

from .APIKeyAuthentication import APIKeyAuthentication
from .CustomJWTAuthentication import CustomJWTAuthentication
from .serializers import BotSerializer
from .models import Bot
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
import os
import PyPDF2
from docx import Document
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated


@api_view(['GET'])
def apiOverview(request):
    api_urls = {
        'List': '/task-list/',
    }

    return Response(api_urls)


@api_view(['GET'])
# @permission_classes([HasAPIKey])
# @authentication_classes([HasAPIKey])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def get_user(request):
    return Response(request.user.__str__())


@api_view(["POST"])
# @permission_classes([HasAPIKey])
@permission_classes([HasAPIKey, JWTAuthentication])
def generate_names(request):
    if request.method == 'POST':
        # Parse the request body and extract the prompt
        prompt = request.data.get('prompt')

        # Set up the OpenAI API client
        openai.api_key = "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"

        # Define a generator function to stream the response
        def generate_response():
            for chunk in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    yield content

        # Return a streaming response to the client
        return StreamingHttpResponse(generate_response(), content_type='text/event-stream')

    # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


@api_view(["POST"])
def falcon(request):
    if request.method == 'POST':
        # Parse the request body and extract the prompt
        prompt = request.data.get('prompt')

        client = Client("https://tiiuae-falcon-chat.hf.space/")
        result = client.predict(
            # "Howdy!",  # str  in 'Click on any example and press Enter in the input textbox!' Dataset component
            prompt,  # str  in 'Click on any example and press Enter in the input textbox!' Dataset component
            fn_index=0
        )
        return Response(result)

    # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# @authentication_classes([JWTAuthentication])
# def init_bot(request):
#     if request.method == 'POST':
#         bot_name = request.POST.get('bot_name')
#
#         # Assuming the bot folder structure is "media/<bot_name>/"
#         base_path = os.path.join('media', bot_name)
#         db_path = os.path.join(base_path, 'db')
#
#         os.makedirs(db_path, exist_ok=True)
#
#         merged_file_path = os.path.join(db_path, 'merged.txt')
#
#         # Retrieve all TXT, PDF, and MS Word files inside the bot folder
#         txt_files = [file for file in os.listdir(base_path) if file.endswith('.txt')]
#         pdf_files = [file for file in os.listdir(base_path) if file.endswith('.pdf')]
#         docx_files = [file for file in os.listdir(base_path) if file.endswith('.docx')]
#
#         # Iterate over the TXT, PDF, and MS Word files and combine their content into merged.txt
#         with open(merged_file_path, 'w') as merged_file:
#             for file_path in txt_files + pdf_files + docx_files:
#                 file_path = os.path.join(base_path, file_path)
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                     merged_file.write(file.read())
#
#         return HttpResponse(f"Bot '{bot_name}' initiated successfully.")@api_view(['POST'])

# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# @authentication_classes([JWTAuthentication])
# def init_bot(request):
#     if request.method == 'POST':
#         bot_name = request.POST.get('bot_name')
#
#         # Assuming the bot folder structure is "media/<bot_name>/"
#         base_path = os.path.join('media', bot_name)
#         db_path = os.path.join(base_path, 'db')
#
#         os.makedirs(db_path, exist_ok=True)
#
#         merged_file_path = os.path.join(db_path, 'merged.txt')
#
#         # Retrieve all TXT, PDF, and MS Word files inside the bot folder
#         txt_files = [file for file in os.listdir(base_path) if file.endswith('.txt')]
#         pdf_files = [file for file in os.listdir(base_path) if file.endswith('.pdf')]
#         docx_files = [file for file in os.listdir(base_path) if file.endswith('.docx')]
#
#         # DEBUG: Print the detected files
#         print(txt_files)
#         print(pdf_files)
#         print(docx_files)
#
#         # Merge file content into merged.txt
#         merged_content = ""
#         for file_path in txt_files + pdf_files + docx_files:
#             file_path = os.path.join(base_path, file_path)
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 file_content = file.read()
#                 merged_content += file_content
#
#         print(merged_content)  # DEBUG: Print the merged content
#
#         # Write merged content to merged.txt
#         with open(merged_file_path, 'w') as merged_file:
#             merged_file.write(merged_content)
#
#         return HttpResponse(f"Bot '{bot_name}' initiated successfully.")

#
# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# @authentication_classes([JWTAuthentication])
# def init_bot(request):
#     if request.method == 'POST':
#         bot_name = request.POST.get('bot_name')
#
#         # Assuming the bot folder structure is "media/<bot_name>/documents/"
#         base_path = os.path.join('media', bot_name)
#         documents_path = os.path.join(base_path, 'documents')
#         db_path = os.path.join(documents_path, 'db')
#         merged_file_path = os.path.join(db_path, 'merged.txt')
#
#         # Retrieve all TXT, PDF, and MS Word files inside the "media/<bot_name>/documents/" folder
#         txt_files = [file for file in os.listdir(documents_path) if file.endswith('.txt')]
#         # pdf_files = [file for file in os.listdir(documents_path) if file.endswith('.pdf')]
#         docx_files = [file for file in os.listdir(documents_path) if file.endswith('.docx')]
#
#         # Merge file content into merged.txt
#         merged_content = ""
#         # for file_path in txt_files + pdf_files + docx_files:
#         for file_path in txt_files + docx_files:
#             file_path = os.path.join(documents_path, file_path)
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 file_content = file.read()
#                 merged_content += file_content
#
#         # Write merged content to merged.txt, overwriting existing content
#         with open(merged_file_path, 'w') as merged_file:
#             merged_file.write(merged_content)
#
#         return HttpResponse(f"Bot '{bot_name}' initiated successfully.")


# @api_view(['POST'])
# @permission_classes([IsAuthenticated])
# @authentication_classes([JWTAuthentication])
# def init_bot(request):
#     if request.method == 'POST':
#         bot_name = request.POST.get('bot_name')
#
#         # Assuming the bot folder structure is "media/<bot_name>/documents/"
#         base_path = os.path.join('media', bot_name)
#         documents_path = os.path.join(base_path, 'documents')
#         db_path = os.path.join(base_path, 'db')
#         merged_file_path = os.path.join(db_path, 'merged.txt')
#
#         # Retrieve all TXT, PDF, and DOCX files inside the "media/<bot_name>/documents/" folder
#         txt_files = [file for file in os.listdir(documents_path) if file.endswith('.txt')]
#         pdf_files = [file for file in os.listdir(documents_path) if file.endswith('.pdf')]
#         docx_files = [file for file in os.listdir(documents_path) if file.endswith('.docx')]
#
#         # Merge file content
#         merged_content = ""
#
#         # Process TXT files
#         for file_path in txt_files:
#             file_path = os.path.join(documents_path, file_path)
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
#                 file_content = file.read()
#                 merged_content += file_content
#
#         # Process PDF files
#         for file_path in pdf_files:
#             file_path = os.path.join(documents_path, file_path)
#             with open(file_path, 'rb') as file:
#                 # pdf = PyPDF2.PdfFileReader(file)
#                 pdf = PyPDF2.PdfReader(file)
#                 for page_num in range(len(pdf.pages)):
#                     # page = pdf.getPage(page_num)
#                     page = pdf.pages[page_num]
#                     page_text = page.extract_text()
#                     merged_content += page_text
#
#         # Process DOCX files
#         for file_path in docx_files:
#             file_path = os.path.join(documents_path, file_path)
#             doc = Document(file_path)
#             for paragraph in doc.paragraphs:
#                 paragraph_text = paragraph.text
#                 merged_content += paragraph_text
#
#         # Write merged content to merged.txt, overwriting existing content
#         with open(merged_file_path, 'w') as merged_file:
#             merged_file.write(merged_content)
#
#         return HttpResponse(f"Bot '{bot_name}' initiated successfully.")


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def init_bot(request):
    if request.method == 'POST':
        bot_name = request.POST.get('bot_name')

        # Assuming the bot folder structure is "media/<bot_name>/documents/"
        base_path = os.path.join('media', bot_name)
        documents_path = os.path.join(base_path, 'documents')
        db_path = os.path.join(base_path, 'db')
        merged_file_path = os.path.join(db_path, 'merged.txt')

        txt_files = [file for file in os.listdir(documents_path) if file.endswith('.txt')]
        pdf_files = [file for file in os.listdir(documents_path) if file.endswith('.pdf')]
        docx_files = [file for file in os.listdir(documents_path) if file.endswith('.docx')]

        merged_content = ""

        for file_path in txt_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
                merged_content += file_content

        for file_path in pdf_files:
            file_path = os.path.join(documents_path, file_path)
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                for page_num in range(len(pdf.pages)):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    merged_content += page_text

        for file_path in docx_files:
            file_path = os.path.join(documents_path, file_path)
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                paragraph_text = paragraph.text
                merged_content += paragraph_text

        with open(merged_file_path, 'w') as merged_file:
            merged_file.write(merged_content)

        # Vectorization
        os.environ["OPENAI_API_KEY"] = "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"
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )

        texts = text_splitter.split_text(merged_content)

        embeddings = OpenAIEmbeddings()
        metadatas = [{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))]
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas,
                                      persist_directory=db_path)  # Changes persist directory
        docsearch.persist()

        return HttpResponse(f"Bot '{bot_name}' initiated successfully.")


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def query_bot(request):
    if request.method == "POST":
        bot_name = request.POST.get('bot_name')
        user_input = request.POST.get('query')

        # Set the OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = "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"

        # Relative path of the bot's document directory
        base_path = os.path.join('media', bot_name)
        db_path = os.path.join(base_path, 'db')

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Load Chroma instance from existing database
        docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Create chain instance with OpenAI and chosen chain_type
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

        # Query the chain
        result = chain({"question": user_input}, return_only_outputs=True)

        # Extract the answer and source information
        answer = result["answer"].replace('\n', ' ')
        source = result["sources"]

        # Return a response in JSON format
        return JsonResponse({
            'bot_name': bot_name,
            'query': user_input,
            'answer': answer,
            'source': source
        })

    return JsonResponse({'error': 'Invalid Method'})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def query_bot_s(request):
    if request.method == "POST":
        bot_name = request.POST.get('bot_name')
        user_input = request.POST.get('query')

        # Set the OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = "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"

        # Relative path of the bot's document directory
        base_path = os.path.join('media', bot_name)
        db_path = os.path.join(base_path, 'db')

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Load Chroma instance from existing database
        docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

        # Create chain instance with OpenAI and chosen chain_type
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff",
                                                            retriever=docsearch.as_retriever())

        # Query the chain
        result = chain({"question": user_input}, return_only_outputs=True)

        # Extract the answer and source information
        answer = result["answer"].replace('\n', ' ')
        source = result["sources"]

        # Set up the OpenAI API client
        openai.api_key = "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"

        prompt = "Context: {} Question: {}"
        # prompt = '''Context: {} Question: {}'''

        sex = prompt.format(answer, user_input)

        # Define a generator function to stream the response
        def generate_response_now():
            for chunk in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": sex
                    }],
                    stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    yield content

        # Return a streaming response to the client
        return StreamingHttpResponse(generate_response_now(), content_type='text/event-stream')

        # Return a JSON error if the request method is not POST
    return JsonResponse({'error': 'Method not allowed.'}, status=405)


class BotViewSet(viewsets.ModelViewSet):
    queryset = Bot.objects.all()
    serializer_class = BotSerializer
    # authentication_classes = [JWTAuthentication, HasAPIKey]
    # authentication_classes = [JWTAuthentication, APIKeyAuthentication]
    authentication_classes = [JWTAuthentication]
    # authentication_classes = [APIKeyAuthentication]
    # authentication_classes = [HasAPIKey]
    permission_classes = [IsAuthenticated]

    # permission_classes = [HasAPIKey]

    def get_queryset(self):
        return Bot.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


# The error occurred because you're treating `HasAPIKey` as an `authentication_class`. `HasAPIKey` is actually a `permission_class`.
#
# This is how it should look:
#
# ```python
# from rest_framework_api_key.models import APIKey
# from rest_framework_api_key.permissions import HasAPIKey
#
# class BotViewSet(viewsets.ModelViewSet):
#     queryset = Bot.objects.all()
#     serializer_class = BotSerializer
#     permission_classes = [IsAuthenticated | HasAPIKey]
#
#     def get_queryset(self):
#         return Bot.objects.filter(user=self.request.user)
#
#     def perform_create(self, serializer):
#         serializer.save(user=self.request.user)
# ```
#
# This way, when Django is doing authentication, it'll check if either of the permission classes (`IsAuthenticated` or `HasAPIKey`) returns `True`. In this scenario, an authenticated JWT token or a correct API key will let the request pass.
#
# However, if you need to associate the user with the API Key, you would have to create a custom Authentication class that extends from `BaseAuthentication` and override the `authenticate` method to get the user from the `APIKey` model.
#
# ```python
# from rest_framework.authentication import BaseAuthentication
# from rest_framework.exceptions import AuthenticationFailed
# from rest_framework_api_key.models import APIKey
#
#
# class APIKeyAuthentication(BaseAuthentication):
#     def authenticate(self, request):
#         api_key_header = request.META.get('HTTP_AUTHORIZATION')
#         if not api_key_header or 'API-Key' not in api_key_header:
#            return None
#         _, api_key_string = api_key_header.split()
#         api_key = APIKey.objects.filter(key=api_key_string).first()
#         if not api_key:
#             raise AuthenticationFailed('Invalid API Key.')
#
#         return (api_key.created_by, api_key)
# ```
#
# You then implement this authentication class in your view:
#
# ```python
# class BotViewSet(viewsets.ModelViewSet):
#     queryset = Bot.objects.all()
#     serializer_class = BotSerializer
#     authentication_classes = [CustomJWTAuthentication, APIKeyAuthentication]
#     permission_classes = [IsAuthenticated]
#
#     def get_queryset(self):
#         return Bot.objects.filter(user=self.request.user)
#
#     def perform_create(self, serializer):
#         serializer.save(user=self.request.user)
# ```
#
# Remember to protect your API Key from being exposed in a production environment as this could lead to potential security issues.
